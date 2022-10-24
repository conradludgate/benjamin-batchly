//! Low latency batching tool.
//! Bundle lots of single concurrent operations into sequential batches of work.
//!
//! For example many concurrent contending single edatabase update tasks could be
//! bundled into bulk updates.
//!
//! # Example
//! ```
//! use benjamin_batchly::{BatchMutex, BatchResult};
//!
//! # async fn example() -> Result<(), &'static str> {
//! let batcher = BatchMutex::default();
//!
//! # let (batch_key, item) = (1, 2);
//! # async fn db_bulk_insert(_: impl IntoIterator<Item=i32>) -> Result<(), &'static str> { Ok(()) }
//! // BatchMutex synchronizes so only one `Work` happens at a time (for a given batch_key).
//! // All concurrent submissions made while an existing `Work` is being processed will
//! // await completion and form the next `Work` batch.
//! match batcher.submit(batch_key, item).await {
//!     BatchResult::Work(mut batch) => {
//!         db_bulk_insert(&mut batch).await?;
//!         batch.notify_all_done();
//!         Ok(())
//!     }
//!     BatchResult::Done(_) => Ok(()),
//!     BatchResult::Failed => Err("failed"),
//! }
//! # }
//! ```
//!
//! # Example: Return values
//! Each item may also received it's own return value inside [`BatchResult::Done`].
//!
//! E.g. a `Result` to pass back why some batch items failed to their submitters.
//! ```
//! use benjamin_batchly::{BatchMutex, BatchResult};
//! use anyhow::anyhow;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // 3rd type is value returned by BatchResult::Done
//! let batcher: BatchMutex<_, _, anyhow::Result<()>> = BatchMutex::default();
//!
//! # let (batch_key, my_item) = (1, 2);
//! # async fn db_bulk_insert(_: impl IntoIterator<Item=i32>) -> Vec<(usize, bool)> { <_>::default() }
//! match batcher.submit(batch_key, my_item).await {
//!     BatchResult::Work(mut batch) => {
//!         let results = db_bulk_insert(&mut batch).await;
//!
//!         // iterate over results and notify each item's submitter
//!         for (index, success) in results {
//!             if success {
//!                 batch.notify_done(index, Ok(()));
//!             } else {
//!                 batch.notify_done(index, Err(anyhow!("insert failed")));
//!             }
//!         }
//!
//!         // receive the local `my_item` return value
//!         batch.finish().unwrap()
//!     }
//!     BatchResult::Done(result) => result,
//!     BatchResult::Failed => Err(anyhow!("batch failed")),
//! }
//! # }
//! ```
use hashbrown::{hash_map::EntryRef, HashMap};
use std::{
    borrow::Borrow,
    collections::hash_map::RandomState,
    hash::{BuildHasher, Hash, Hasher},
    sync::{Arc, Mutex, MutexGuard},
};
use std::{fmt, mem};
use tokio::sync::{Semaphore, SemaphorePermit};

/// Batch HQ. Share and use concurrently to dynamically batch submitted items.
///
/// Cheap to clone (`Arc` guts).
///
/// See [`BatchMutex::submit`] & crate docs.
///
/// * `Key` batch key type.
/// * `Item` single item type to be batched together into a `Vec<Item>`.
/// * `T` value returned by [`BatchResult::Done`], default `()`.
pub struct BatchMutex<Key, Item, T = ()> {
    hasher: RandomState,
    queue: Arc<[Shard<Key, Item, T>]>,
}

impl<Key, Item, T> Clone for BatchMutex<Key, Item, T> {
    fn clone(&self) -> Self {
        Self {
            hasher: self.hasher.clone(),
            queue: self.queue.clone(),
        }
    }
}

impl<Key: Eq + Hash, Item, T> Default for BatchMutex<Key, Item, T> {
    fn default() -> Self {
        // this is the same default shard size as DashMap.
        // we are free to tinker with this and make it customisable too
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        let cap = threads.next_power_of_two() * 4;
        let mut shards = Vec::with_capacity(cap);

        let hasher = RandomState::new();
        shards.resize_with(cap, || Mutex::new(HashMap::with_hasher(hasher.clone())));

        Self {
            hasher,
            queue: shards.into_boxed_slice().into(),
        }
    }
}

impl<Key, Item, T> fmt::Debug for BatchMutex<Key, Item, T>
where
    Key: Eq + Hash,
    Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchMutex").finish_non_exhaustive()
    }
}

impl<Key, Item, T> BatchMutex<Key, Item, T>
where
    Key: Eq + Hash,
{
    fn get_shard(&self, key: &Key) -> ShardGuard<Key, Item, T> {
        let mut hasher = self.hasher.build_hasher();
        key.hash(&mut hasher);
        // queue len is always a power of two, so it should evenly divide the hash space and be fair
        let index = (hasher.finish()) as usize % self.queue.len();
        self.queue[index].lock().unwrap()
    }
}

impl<Key, Item, T> BatchMutex<Key, Item, T>
where
    Key: Eq + Hash + Clone,
{
    /// Submits an `item` to be processed as a batch, ie with other items
    /// submitted with the same `batch_key`.
    ///
    /// Submissions made while a previous `batch_key` batch is processing will await
    /// that batch finishing. All such waiting items become the next batch as soon
    /// as previous finishes.
    ///
    /// Note since submission adds no artificial latency the very first call will
    /// always result in a batch of a single item.
    ///
    /// # Example
    /// ```
    /// # use benjamin_batchly::{BatchMutex, BatchResult};
    /// # async fn example() -> Result<(), &'static str> {
    /// # let batcher = BatchMutex::default();
    /// # let (batch_key, item) = (1, 2);
    /// # async fn db_bulk_insert(_: impl IntoIterator<Item=i32>) -> Result<(), &'static str> { Ok(()) }
    /// // Synchronizes so only one `Work` happens at a time (for a given batch_key).
    /// // All concurrent submissions made while an existing `Work` is being processed will
    /// // await completion and form the next `Work` batch.
    /// match batcher.submit(batch_key, item).await {
    ///     BatchResult::Work(mut batch) => {
    ///         db_bulk_insert(&mut batch).await?;
    ///         batch.notify_all_done();
    ///         Ok(())
    ///     }
    ///     BatchResult::Done(_) => Ok(()),
    ///     BatchResult::Failed => Err("failed"),
    /// }
    /// # }
    /// ```
    pub async fn submit(&self, batch_key: Key, item: Item) -> BatchResult<Key, Item, T> {
        let idx;
        let shared_state;
        {
            let mut shard = self.get_shard(&batch_key);
            let state = shard.entry_ref(&batch_key).or_default();
            shared_state = Arc::clone(&state.shared);
            idx = state.items.len();
            state.items.push(ItemState::Ready(item));
        };

        let guard = shared_state
            .permit
            .acquire()
            .await
            .expect("we never close the semaphore so this should never error");
        guard.forget();

        if idx == 0 {
            BatchResult::Work(Batch {
                queue: self.clone(),
                batch_key,
            })
        } else {
            match mem::replace(
                &mut shared_state.items.lock().unwrap()[idx],
                ItemState::Pending,
            ) {
                ItemState::Done(t) => BatchResult::Done(t),
                _ => BatchResult::Failed,
            }
        }
    }
}

struct BatchState<Item, T> {
    items: Vec<ItemState<Item, T>>,
    shared: Arc<BatchSharedState<Item, T>>,
}

struct BatchSharedState<Item, T> {
    items: Mutex<Vec<ItemState<Item, T>>>,
    permit: Semaphore,
}

impl<Item, T> Default for BatchState<Item, T> {
    fn default() -> Self {
        Self {
            items: <_>::default(),
            shared: Arc::new(BatchSharedState {
                items: <_>::default(),
                permit: Semaphore::new(1),
            }),
        }
    }
}

/// [`BatchMutex::submit`] output.
/// Either a batch of items to process & notify or the result of another submitter
/// handling a batch with the same `batch_key`.
#[must_use]
#[derive(Debug)]
pub enum BatchResult<Key: Eq + Hash + Clone, Item, T> {
    /// A non-empty batch of items including the one submitted locally.
    ///
    /// All [`Batch::items`] should be processed and their submitters notified.
    /// See [`Batch::notify_done`], [`Batch::notify_all_done`].
    Work(Batch<Key, Item, T>),
    /// The submitted item has been processed in a batch by another submitter
    /// which has notified the item as done.
    Done(T),
    /// The item started batch processing by another submitter but was dropped
    /// before notifying done.
    Failed,
}

enum ItemState<Item, T> {
    Ready(Item),
    Pending,
    Done(T),
}

/// A batch of items to process.
pub struct Batch<Key: Eq + Hash, Item, T> {
    queue: BatchMutex<Key, Item, T>,
    batch_key: Key,
}

impl<Key, Item, T> fmt::Debug for Batch<Key, Item, T>
where
    Key: Eq + Hash,
    Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Batch").finish_non_exhaustive()
    }
}

impl<'a, Key: Eq + Hash, Item, T> IntoIterator for &'a mut Batch<Key, Item, T> {
    type Item = Item;
    type IntoIter = BatchIter<Item, T>;
    fn into_iter(self) -> Self::IntoIter {
        let mut shard = self.queue.get_shard(&self.batch_key);
        let entry = shard.get_mut(&self.batch_key).unwrap();

        let shared = entry.shared.clone();
        let mut output = shared.items.lock().unwrap();

        // take from the items now
        mem::swap(&mut *output, &mut entry.items);
        let range = 0..output.len();

        drop(shard);
        drop(output);

        BatchIter { shared, range }
    }
}

pub struct BatchIter<Item, T> {
    shared: Arc<BatchSharedState<Item, T>>,
    range: std::ops::Range<usize>,
}

impl<Item, T> Iterator for BatchIter<Item, T> {
    type Item = Item;

    fn next(&mut self) -> Option<Self::Item> {
        let mut output = self.shared.items.lock().unwrap();
        for index in &mut self.range {
            let state = &mut output[index];
            match mem::replace(state, ItemState::Pending) {
                ItemState::Ready(item) => return Some(item),
                state => output[index] = state,
            }
        }
        None
    }
}

impl<Key: Eq + Hash, Item, T> Batch<Key, Item, T> {
    /// Notify an individual item as done.
    ///
    /// This means the submitter of that item will receive a [`BatchResult::Done`] with the done value.
    ///
    /// Note `item_index` is the original index of the item in [`Batch::items`].
    ///
    /// If the item is the local item, the one submitted by the same caller handling the batch,
    /// the result may be received using [`Batch::recv_local_notify_done`].
    pub fn notify_done(&mut self, item_index: usize, val: T) {
        let mut shard = self.queue.get_shard(&self.batch_key);
        let entry = shard.get_mut(&self.batch_key).unwrap();
        let mut output = entry.shared.items.lock().unwrap();
        output
            .get_mut(item_index)
            .map(|i| mem::replace(i, ItemState::Done(val)));
    }

    /// Receive the local item's done notification if available.
    ///
    /// The local item is the one submitted by the caller that received [`BatchResult::Work`].
    ///
    /// Since the local item is handled along with multiple other non-local items, this method
    /// can be used to get the local computation result after handling all items and calling
    /// [`Batch::notify_done`] for each (one of which is the local item).
    pub fn finish(self) -> Option<T> {
        let mut shard = self.queue.get_shard(&self.batch_key);
        let entry = shard.get_mut(&self.batch_key).unwrap();
        let mut output = entry.shared.items.lock().unwrap();
        match mem::replace(&mut output[0], ItemState::Pending) {
            ItemState::Ready(_) | ItemState::Pending => None,
            ItemState::Done(val) => Some(val),
        }
    }

    // /// Takes items that have been submitted after this [`Batch`] was returned and adds
    // /// them to this batch to be processed immediately.
    // ///
    // /// New items are appended onto [`Batch::items`].
    // ///
    // /// Returns `true` if any new items were pulled in.
    // pub fn pull_waiting_items(&mut self) -> bool {
    //     let mut shard = self.queue.get_shard(&self.batch_key);
    //     match shard.get_mut(&self.batch_key) {
    //         Some(next) if !next.items.is_empty() => {
    //             self.items.append(&mut next.items);
    //             true
    //         }
    //         _ => false,
    //     }
    // }
}

impl<Key: Eq + Hash + Clone, Item> Batch<Key, Item, ()> {
    /// Notify all items, that have not already been notified, as done.
    ///
    /// This means all other submitters will receive [`BatchResult::Done`].
    ///
    /// Convenience method when using no/`()` item return value.
    pub fn notify_all_done(&mut self) {
        let mut shard = self.queue.get_shard(&self.batch_key);
        let entry = shard.get_mut(&self.batch_key).unwrap();
        let mut output = entry.shared.items.lock().unwrap();
        for item in output.iter_mut() {
            let _ = mem::replace(item, ItemState::Done(()));
        }
    }
}

impl<Key: Eq + Hash, Item, T> Drop for Batch<Key, Item, T> {
    fn drop(&mut self) {
        // try to cleanup leftover states if possible
        let mut shard = self.queue.get_shard(&self.batch_key);
        if let EntryRef::Occupied(mut entry) = shard.entry_ref(&self.batch_key) {
            let len = entry.get_mut().shared.items.lock().unwrap().len();
            entry.get().shared.permit.add_permits(len);

            // remove entry if no one else is waiting for this permit
            if entry
                .get_mut()
                .shared
                .permit
                .try_acquire()
                .map(SemaphorePermit::forget)
                .is_ok()
            {
                entry.remove();
            }
        }
    }
}

/// Key wrapper. For use with `entry_ref`
/// which uses `From<&K>` to generate the key on insert
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key<K>(K);

impl<K: Clone> From<&'_ K> for Key<K> {
    fn from(k: &'_ K) -> Self {
        Self(k.clone())
    }
}

impl<K> Borrow<K> for Key<K> {
    fn borrow(&self) -> &K {
        &self.0
    }
}

type Shard<K, I, T> = Mutex<HashMap<Key<K>, BatchState<I, T>, RandomState>>;
type ShardGuard<'a, K, I, T> = MutexGuard<'a, HashMap<Key<K>, BatchState<I, T>, RandomState>>;
