//! Concatenate fixed-size arrays at compile time without copying.
//!
//! The easiest entry point is the [`concat_arrays!`] macro:
//!
//! ```rust
//! use const_array_concat::concat_arrays;
//!
//! let arr = concat_arrays!([1u8, 2], [3u8, 4], [5u8, 6]);
//! assert_eq!(arr.as_ref(), &[1, 2, 3, 4, 5, 6]);
//! ```
//!
//! # How it works
//!
//! [`ArrayConcat<T, A, B>`](ArrayConcat) is a `#[repr(C)]` struct holding a
//! zero-length `[T; 0]` sentinel followed by `A` then `B`.  `repr(C)`
//! guarantees field order and no inter-field padding, so a raw pointer to the
//! sentinel spanning `A::SIZE + B::SIZE` elements covers both arrays as one
//! contiguous slice — no copy ever happens.
//!
//! # Building a concatenation
//!
//! Use [`ArrayConcat::new`] and chain [`append`], [`prepend`], or [`insert`],
//! then optionally collapse to a plain array with [`into_array`]:
//!
//! ```rust
//! use const_array_concat::ArrayConcat;
//!
//! let arr: [u8; 6] = ArrayConcat::new([1u8, 2], [5u8, 6])
//!     .insert([3u8, 4])
//!     .into_array();
//!
//! assert_eq!(arr, [1, 2, 3, 4, 5, 6]);
//! ```
//!
//! [`append`]: ArrayConcat::append
//! [`prepend`]: ArrayConcat::prepend
//! [`insert`]: ArrayConcat::insert
//! [`into_array`]: ArrayConcat::into_array

#![no_std]

use core::mem::{self, ManuallyDrop};

use const_panic::concat_assert;

/// Concatenate two or more arrays into an [`ArrayConcat`].
///
/// - **Zero arguments** — returns an empty array `[]`.
/// - **One argument** — returns the argument unchanged.
/// - **Two or more arguments** — expands to `ArrayConcat::new([], first).append(second)…`,
///   producing a left-nested [`ArrayConcat`] with a `[T; 0]` sentinel on the far left.
///
/// Trailing commas are accepted.
///
/// # Examples
///
/// ```rust
/// use const_array_concat::{concat_arrays, ConcatableArray};
///
/// let arr = concat_arrays!([1u8, 2], [3u8, 4]);
/// assert_eq!(arr.as_ref(), &[1u8, 2, 3, 4]);
///
/// let arr = concat_arrays!([1u8, 2], [3u8, 4], [5u8, 6]);
/// assert_eq!(arr.as_ref(), &[1u8, 2, 3, 4, 5, 6]);
///
/// // Single array passes through unchanged
/// assert_eq!(concat_arrays!([7u8, 8, 9]), [7u8, 8, 9]);
///
/// // No arguments produces an empty array
/// let empty: [u8; 0] = concat_arrays!();
/// assert_eq!(empty, []);
/// ```
#[macro_export]
macro_rules! concat_arrays {
    () => {[]};
    ($v:expr) => {$v};
    ($a:expr, $($rest:expr),+ $(,)?) => {
        $crate::ArrayConcat::new([], $a)
        $(.append($rest))+
    };
}

/// Marker trait for arrays that can participate in a concatenation.
///
/// # Safety
///
/// The implementing type must have no padding — its in-memory representation
/// must be exactly `SIZE * size_of::<T>()` bytes with no leading, trailing, or
/// interior padding bytes.  [`ArrayConcat`] relies on this when it reads across
/// the boundary between two adjacent implementors as a single flat slice.
pub unsafe trait ConcatableArray: AsRef<[Self::T]> + AsMut<[Self::T]> {
    /// The element type.
    type T;
    /// The number of elements.
    const SIZE: usize;
}

// SAFETY: [T; N] is exactly N * size_of::<T>() bytes with no padding, which
// satisfies the no-padding requirement of ConcatableArray.
unsafe impl<T, const N: usize> ConcatableArray for [T; N] {
    type T = T;
    const SIZE: usize = N;
}

/// Two [`ConcatableArray`]s laid out contiguously in memory.
///
/// The struct is `#[repr(C)]` with a leading `[T; 0]` sentinel, followed by
/// `A` then `B`.  This guarantees that `A`'s first element sits at the same
/// address as the sentinel, so a single `from_raw_parts` call spanning
/// `A::SIZE + B::SIZE` elements covers both arrays without any copy.
///
/// Use [`new`](Self::new) to construct, and [`as_ref`](AsRef::as_ref) /
/// [`as_mut`](AsMut::as_mut) to access the combined slice.
#[repr(C)]
pub struct ArrayConcat<T, A, B>([T; 0], A, B);

impl<T, A, B> ArrayConcat<T, A, B>
where
    A: ConcatableArray<T = T>,
    B: ConcatableArray<T = T>,
{
    /// Construct a concatenation of `a` followed by `b`.
    ///
    /// ```rust
    /// use const_array_concat::ArrayConcat;
    ///
    /// let arr = ArrayConcat::new([1u8, 2, 3], [4u8, 5]);
    /// assert_eq!(arr.as_ref(), &[1, 2, 3, 4, 5]);
    /// ```
    pub const fn new(a: A, b: B) -> Self {
        const {
            concat_assert!(mem::size_of::<Self>() == mem::size_of::<A>() + mem::size_of::<B>());
        }
        Self([], a, b)
    }

    /// Split the concatenation back into its two constituent arrays.
    ///
    /// Both halves are returned as [`ManuallyDrop`] to avoid a double-drop of
    /// the original value, which is consumed via [`mem::forget`].
    pub const fn decompose(v: ManuallyDrop<Self>) -> (ManuallyDrop<A>, ManuallyDrop<B>) {
        let v = ManuallyDrop::into_inner(v);
        // SAFETY: v.1 and v.2 are valid, properly aligned, and initialized.
        // mem::forget below prevents v from dropping them, so no double-free.
        let a = unsafe { core::ptr::read(&v.1) };
        let b = unsafe { core::ptr::read(&v.2) };
        mem::forget(v);
        (ManuallyDrop::new(a), ManuallyDrop::new(b))
    }

    /// Append `c` after `self`, producing `[…self…, …c…]`.
    ///
    /// ```rust
    /// use const_array_concat::ArrayConcat;
    ///
    /// let arr = ArrayConcat::new([1u8, 2], [3u8, 4])
    ///     .append([5u8, 6]);
    /// assert_eq!(arr.as_ref(), &[1, 2, 3, 4, 5, 6]);
    /// ```
    pub const fn append<C: ConcatableArray<T = T>>(self, c: C) -> ArrayConcat<T, Self, C> {
        ArrayConcat::new(self, c)
    }

    /// Prepend `c` before `self`, producing `[…c…, …self…]`.
    ///
    /// ```rust
    /// use const_array_concat::ArrayConcat;
    ///
    /// let arr = ArrayConcat::new([3u8, 4], [5u8, 6])
    ///     .prepend([1u8, 2]);
    /// assert_eq!(arr.as_ref(), &[1, 2, 3, 4, 5, 6]);
    /// ```
    pub const fn prepend<C: ConcatableArray<T = T>>(self, c: C) -> ArrayConcat<T, C, Self> {
        ArrayConcat::new(c, self)
    }

    /// Insert `c` between `A` and `B`, producing `[…A…, …c…, …B…]`.
    ///
    /// ```rust
    /// use const_array_concat::ArrayConcat;
    ///
    /// let arr = ArrayConcat::new([1u8, 2], [5u8, 6])
    ///     .insert([3u8, 4]);
    /// assert_eq!(arr.as_ref(), &[1, 2, 3, 4, 5, 6]);
    /// ```
    pub const fn insert<C: ConcatableArray<T = T>>(
        self,
        c: C,
    ) -> ArrayConcat<T, ArrayConcat<T, A, C>, B> {
        let (a, b) = Self::decompose(ManuallyDrop::new(self));
        ArrayConcat::new(ManuallyDrop::into_inner(a), c).append(ManuallyDrop::into_inner(b))
    }

    /// Convert the concatenation into a plain array of exactly `N` elements.
    ///
    /// `N` must equal the total [`SIZE`](ConcatableArray::SIZE); a mismatch is
    /// a compile-time error.
    ///
    /// ```rust
    /// use const_array_concat::ArrayConcat;
    ///
    /// let arr: [u8; 5] = ArrayConcat::new([1u8, 2, 3], [4u8, 5]).into_array();
    /// assert_eq!(arr, [1, 2, 3, 4, 5]);
    /// ```
    ///
    /// Passing the wrong size fails to compile:
    ///
    /// ```rust,compile_fail
    /// use const_array_concat::ArrayConcat;
    ///
    /// // SIZE is 5 but N is 4 — compile error
    /// let _: [u8; 4] = ArrayConcat::new([1u8, 2, 3], [4u8, 5]).into_array();
    /// ```
    pub const fn into_array<const N: usize>(self) -> [T; N] {
        const {
            concat_assert!(N == Self::SIZE);
        }
        let src = ManuallyDrop::new(self);
        // SAFETY: Self is #[repr(C)] with no padding (ConcatableArray invariant),
        // so its SIZE * size_of::<T>() bytes are identical in layout to [T; N].
        // The const assert above guarantees N == Self::SIZE, so the sizes match.
        // ManuallyDrop prevents a double-drop of src after the bit-copy.
        unsafe { mem::transmute_copy(&src) }
    }
}

impl<T, A, B> AsRef<[T]> for ArrayConcat<T, A, B>
where
    A: ConcatableArray<T = T>,
    B: ConcatableArray<T = T>,
{
    fn as_ref(&self) -> &[T] {
        // SAFETY: self.0 points to the start of the #[repr(C)] struct, which
        // is immediately followed by A then B with no padding (ConcatableArray
        // invariant). The resulting slice lives as long as &self.
        unsafe { core::slice::from_raw_parts(self.0.as_ptr(), Self::SIZE) }
    }
}
impl<T, A, B> AsMut<[T]> for ArrayConcat<T, A, B>
where
    A: ConcatableArray<T = T>,
    B: ConcatableArray<T = T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        // SAFETY: same layout argument as as_ref; exclusive &mut self guarantees
        // no aliasing for the duration of the returned slice.
        unsafe { core::slice::from_raw_parts_mut(self.0.as_mut_ptr(), Self::SIZE) }
    }
}

// SAFETY: ArrayConcat is #[repr(C)] with a [T; 0] sentinel, A, then B. Both A
// and B have no padding (ConcatableArray invariant), and repr(C) adds no
// padding between fields of the same alignment, so the struct itself has no
// padding and is exactly (A::SIZE + B::SIZE) * size_of::<T>() bytes.
unsafe impl<T, A, B> ConcatableArray for ArrayConcat<T, A, B>
where
    A: ConcatableArray<T = T>,
    B: ConcatableArray<T = T>,
{
    type T = T;
    const SIZE: usize = A::SIZE + B::SIZE;
}

/// Assert at compile time that these types have the same size.
///
/// This is an implementation detail of the crate and should only be used by the
/// macros in this crate.
#[inline(always)]
#[doc(hidden)]
pub const fn _const_assert_same_size<A, B>() -> bool {
    const { core::mem::size_of::<A>() == core::mem::size_of::<B>() || panic!("Size Mismatch") }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Concat<T, const N: usize, const M: usize> = ArrayConcat<T, [T; N], [T; M]>;

    #[test]
    fn size_is_sum_of_parts() {
        assert_eq!(<Concat<u8, 3, 2> as ConcatableArray>::SIZE, 5);
        assert_eq!(<Concat<u32, 0, 4> as ConcatableArray>::SIZE, 4);
        assert_eq!(<Concat<u8, 0, 0> as ConcatableArray>::SIZE, 0);
    }

    #[test]
    fn as_ref_yields_concatenated_elements() {
        let arr: Concat<u8, 3, 2> = ArrayConcat::new([1, 2, 3], [4, 5]);
        assert_eq!(arr.as_ref(), &[1u8, 2, 3, 4, 5]);
    }

    #[test]
    fn as_mut_allows_writing_across_boundary() {
        let mut arr: Concat<i32, 2, 2> = ArrayConcat::new([10, 20], [30, 40]);
        arr.as_mut()[1] = 99;
        arr.as_mut()[2] = 88;
        assert_eq!(arr.as_ref(), &[10, 99, 88, 40]);
    }

    #[test]
    fn empty_left_array() {
        let arr: Concat<u8, 0, 3> = ArrayConcat::new([], [7, 8, 9]);
        assert_eq!(arr.as_ref(), &[7u8, 8, 9]);
    }

    #[test]
    fn empty_right_array() {
        let arr: Concat<u8, 3, 0> = ArrayConcat::new([1, 2, 3], []);
        assert_eq!(arr.as_ref(), &[1u8, 2, 3]);
    }

    #[test]
    fn append_chains_to_right() {
        type Inner = ArrayConcat<u8, [u8; 2], [u8; 2]>;
        type Outer = ArrayConcat<u8, Inner, [u8; 2]>;
        let arr: Outer = ArrayConcat::new([1, 2], [3, 4]).append([5, 6]);
        assert_eq!(arr.as_ref(), &[1u8, 2, 3, 4, 5, 6]);
        assert_eq!(<Outer as ConcatableArray>::SIZE, 6);
    }

    #[test]
    fn prepend_chains_to_left() {
        type Inner = ArrayConcat<u8, [u8; 2], [u8; 2]>;
        type Outer = ArrayConcat<u8, [u8; 2], Inner>;
        let arr: Outer = ArrayConcat::new([3, 4], [5, 6]).prepend([1, 2]);
        assert_eq!(arr.as_ref(), &[1u8, 2, 3, 4, 5, 6]);
        assert_eq!(<Outer as ConcatableArray>::SIZE, 6);
    }

    #[test]
    fn insert_places_between_a_and_b() {
        // [1,2] ++ [5,6] with [3,4] inserted => [1,2,3,4,5,6]
        type Base = ArrayConcat<u8, [u8; 2], [u8; 2]>;
        type Result = ArrayConcat<u8, ArrayConcat<u8, [u8; 2], [u8; 2]>, [u8; 2]>;
        let arr: Result = ArrayConcat::new([1, 2], [5, 6]).insert([3, 4]);
        assert_eq!(arr.as_ref(), &[1u8, 2, 3, 4, 5, 6]);
        assert_eq!(<Result as ConcatableArray>::SIZE, 6);
        // verify B is unchanged by checking tail
        let _ = <Base as ConcatableArray>::SIZE; // Base still compiles fine
    }

    #[test]
    fn insert_empty_is_identity() {
        type Result = ArrayConcat<u8, ArrayConcat<u8, [u8; 2], [u8; 0]>, [u8; 2]>;
        let arr: Result = ArrayConcat::new([1, 2], [3, 4]).insert([]);
        assert_eq!(arr.as_ref(), &[1u8, 2, 3, 4]);
    }

    #[test]
    fn into_array_correct_size() {
        let arr: [u8; 5] = ArrayConcat::new([1u8, 2, 3], [4u8, 5]).into_array();
        assert_eq!(arr, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn into_array_chained() {
        let arr: [u8; 6] = ArrayConcat::new([1u8, 2], [3u8, 4])
            .append([5u8, 6])
            .into_array();
        assert_eq!(arr, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn into_array_preserves_values_across_boundary() {
        let mut concat = ArrayConcat::new([10i32, 20], [30i32, 40]);
        concat.as_mut()[1] = 99;
        let arr: [i32; 4] = concat.into_array();
        assert_eq!(arr, [10, 99, 30, 40]);
    }

    #[test]
    fn macro_empty() {
        let arr: [u8; 0] = concat_arrays!();
        assert_eq!(arr, []);
    }

    #[test]
    fn macro_single() {
        assert_eq!(concat_arrays!([1u8, 2, 3]), [1u8, 2, 3]);
    }

    #[test]
    fn macro_two() {
        let arr = concat_arrays!([1u8, 2], [3u8, 4]);
        assert_eq!(arr.as_ref(), &[1u8, 2, 3, 4]);
    }

    #[test]
    fn macro_three() {
        let arr = concat_arrays!([1u8, 2], [3u8, 4], [5u8, 6]);
        assert_eq!(arr.as_ref(), &[1u8, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn macro_trailing_comma() {
        let arr = concat_arrays!([1u8, 2], [3u8, 4],);
        assert_eq!(arr.as_ref(), &[1u8, 2, 3, 4]);
    }
}
