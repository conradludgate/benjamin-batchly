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
//! # async fn db_bulk_insert(_: &[i32]) -> Result<(), &'static str> { Ok(()) }
//! // BatchMutex synchronizes so only one `Work` happens at a time (for a given batch_key).
//! // All concurrent submissions made while an existing `Work` is being processed will
//! // await completion and form the next `Work` batch.
//! match batcher.submit(batch_key, item).await {
//!     BatchResult::Work(mut batch) => {
//!         db_bulk_insert(&batch.items).await?;
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
//! # async fn db_bulk_insert(_: &[i32]) -> Vec<(usize, bool)> { <_>::default() }
//! match batcher.submit(batch_key, my_item).await {
//!     BatchResult::Work(mut batch) => {
//!         let results = db_bulk_insert(&batch.items).await;
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
//!         batch.recv_local_notify_done().unwrap()
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
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};
use tokio::{
    pin,
    sync::{oneshot, AcquireError, OwnedSemaphorePermit, Semaphore},
};

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
    /// # async fn db_bulk_insert(_: &[i32]) -> Result<(), &'static str> { Ok(()) }
    /// // Synchronizes so only one `Work` happens at a time (for a given batch_key).
    /// // All concurrent submissions made while an existing `Work` is being processed will
    /// // await completion and form the next `Work` batch.
    /// match batcher.submit(batch_key, item).await {
    ///     BatchResult::Work(mut batch) => {
    ///         db_bulk_insert(&batch.items).await?;
    ///         batch.notify_all_done();
    ///         Ok(())
    ///     }
    ///     BatchResult::Done(_) => Ok(()),
    ///     BatchResult::Failed => Err("failed"),
    /// }
    /// # }
    /// ```
    pub async fn submit(&self, batch_key: Key, item: Item) -> BatchResult<Key, Item, T> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let batch_lock = {
            let mut shard = self.get_shard(&batch_key);
            let state = shard.entry_ref(&batch_key).or_default();
            state.items.push(item);
            state.senders.push(tx);
            Arc::clone(&state.lock)
        };

        let lock = batch_lock.acquire_owned();
        pin!(lock);

        match WorkPermitSelect::new(rx, lock).await {
            WorkPermit::Result(Some(val)) => BatchResult::Done(val),
            WorkPermit::Result(None) => BatchResult::Failed,
            WorkPermit::Permit(guard, rx) => {
                if let Some(guard) = guard {
                    // we return the permit in `Batch::drop`
                    guard.forget()
                }
                let batch = {
                    let mut shard = self.get_shard(&batch_key);
                    let state = shard.get_mut(&batch_key).unwrap(); // should always exist in queue at this point
                    Batch {
                        items: mem::take(&mut state.items),
                        senders: state.senders.drain(..).map(Some).collect(),
                        queue: self.clone(),
                        batch_key,
                        local_rx: rx,
                    }
                };
                BatchResult::Work(batch)
            }
        }
    }
}

struct BatchState<Item, T> {
    items: Vec<Item>,
    senders: Vec<Sender<T>>,
    lock: Arc<Semaphore>,
}

type Sender<T> = oneshot::Sender<Option<T>>;
type Receiver<T> = oneshot::Receiver<Option<T>>;

impl<Item, T> Default for BatchState<Item, T> {
    fn default() -> Self {
        Self {
            items: <_>::default(),
            senders: <_>::default(),
            lock: Arc::new(Semaphore::new(1)),
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

/// A batch of items to process.
pub struct Batch<Key: Eq + Hash, Item, T> {
    /// Batch items.
    pub items: Vec<Item>,
    senders: Vec<Option<Sender<T>>>,
    queue: BatchMutex<Key, Item, T>,
    batch_key: Key,
    local_rx: Receiver<T>,
}

impl<Key, Item, T> fmt::Debug for Batch<Key, Item, T>
where
    Key: Eq + Hash,
    Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Batch")
            .field("items", &self.items)
            .finish_non_exhaustive()
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
        if let Some(tx) = self.senders.get_mut(item_index).and_then(|i| i.take()) {
            let _ = tx.send(Some(val));
        }
    }

    /// Receive the local item's done notification if available.
    ///
    /// The local item is the one submitted by the caller that received [`BatchResult::Work`].
    ///
    /// Since the local item is handled along with multiple other non-local items, this method
    /// can be used to get the local computation result after handling all items and calling
    /// [`Batch::notify_done`] for each (one of which is the local item).
    pub fn recv_local_notify_done(&mut self) -> Option<T> {
        self.local_rx.try_recv().ok().flatten()
    }

    /// Takes items that have been submitted after this [`Batch`] was returned and adds
    /// them to this batch to be processed immediately.
    ///
    /// New items are appended onto [`Batch::items`].
    ///
    /// Returns `true` if any new items were pulled in.
    pub fn pull_waiting_items(&mut self) -> bool {
        let mut shard = self.queue.get_shard(&self.batch_key);
        match shard.get_mut(&self.batch_key) {
            Some(next) if !next.items.is_empty() => {
                self.items.append(&mut next.items);
                self.senders.extend(next.senders.drain(..).map(Some));
                true
            }
            _ => false,
        }
    }

    fn notify_all_failed(&mut self) {
        for tx in &mut self.senders {
            let _ = tx.take().map(|tx| tx.send(None));
        }
    }
}

impl<Key: Eq + Hash + Clone, Item> Batch<Key, Item, ()> {
    /// Notify all items, that have not already been notified, as done.
    ///
    /// This means all other submitters will receive [`BatchResult::Done`].
    ///
    /// Convenience method when using no/`()` item return value.
    pub fn notify_all_done(&mut self) {
        for tx in &mut self.senders {
            let _ = tx.take().map(|tx| tx.send(Some(())));
        }
    }
}

impl<Key: Eq + Hash, Item, T> Drop for Batch<Key, Item, T> {
    fn drop(&mut self) {
        self.notify_all_failed();
        // try to cleanup leftover states if possible
        let mut shard = self.queue.get_shard(&self.batch_key);
        if let EntryRef::Occupied(entry) = shard.entry_ref(&self.batch_key) {
            if Arc::strong_count(&entry.get().lock) == 1 {
                entry.remove();
            } else {
                // return the permit
                entry.get().lock.add_permits(1);
            }
        }
    }
}

enum WorkPermit<T> {
    Result(Option<T>),
    Permit(Option<OwnedSemaphorePermit>, Receiver<T>),
}

struct WorkPermitSelect<'a, T, Acquire> {
    inner: Option<(Receiver<T>, Pin<&'a mut Acquire>)>,
}

impl<'a, T, Acquire> WorkPermitSelect<'a, T, Acquire> {
    fn new(rx: Receiver<T>, acq: Pin<&'a mut Acquire>) -> Self {
        Self {
            inner: Some((rx, acq)),
        }
    }
}

impl<T, Acquire> Future for WorkPermitSelect<'_, T, Acquire>
where
    Acquire: Future<Output = Result<OwnedSemaphorePermit, AcquireError>>,
{
    type Output = WorkPermit<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let (mut a, mut b) = self.inner.take().expect("cannot poll Select twice");

        if let Poll::Ready(val) = Pin::new(&mut a).poll(cx) {
            return Poll::Ready(WorkPermit::Result(val.ok().flatten()));
        }

        if let Poll::Ready(val) = b.as_mut().poll(cx) {
            return Poll::Ready(WorkPermit::Permit(val.ok(), a));
        }

        self.inner = Some((a, b));
        Poll::Pending
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
