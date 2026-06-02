# const-array-concat

Concatenate fixed-size arrays at compile time without copying.

`ArrayConcat<T, A, B>` is a `#[repr(C)]` struct that places two arrays
contiguously in memory so they can be addressed as a single slice — in `const`
contexts, with no heap allocation and no `std`.

## How it works

The struct holds a zero-length `[T; 0]` sentinel followed by `A` then `B`.
`repr(C)` guarantees field order and no inter-field padding, so a raw pointer
to the sentinel spanning `A::SIZE + B::SIZE` elements covers both arrays as one
contiguous slice — no copy ever happens.

## Usage

The easiest entry point is the `concat_arrays!` macro:

```rust
use const_array_concat::concat_arrays;

let arr = concat_arrays!([1u8, 2], [3u8, 4], [5u8, 6]);
assert_eq!(arr.as_ref(), &[1, 2, 3, 4, 5, 6]);
```

Or build up a concatenation manually with `ArrayConcat::new` and the chainable
methods:

```rust
use const_array_concat::ArrayConcat;

// append — add to the right
let arr = ArrayConcat::new([1u8, 2], [3u8, 4])
    .append([5u8, 6]);
assert_eq!(arr.as_ref(), &[1, 2, 3, 4, 5, 6]);

// prepend — add to the left
let arr = ArrayConcat::new([3u8, 4], [5u8, 6])
    .prepend([1u8, 2]);
assert_eq!(arr.as_ref(), &[1, 2, 3, 4, 5, 6]);

// insert — add between the two halves
let arr = ArrayConcat::new([1u8, 2], [5u8, 6])
    .insert([3u8, 4]);
assert_eq!(arr.as_ref(), &[1, 2, 3, 4, 5, 6]);
```

All methods are `const fn`, so concatenations can be built in `const` and
`static` contexts.

Once the shape is final, convert to a plain array with `into_array`. The size
`N` must equal the total length — a mismatch is a **compile-time error**:

```rust
use const_array_concat::concat_arrays;

let arr: [u8; 6] = concat_arrays!([1u8, 2], [3u8, 4], [5u8, 6]).into_array();
assert_eq!(arr, [1, 2, 3, 4, 5, 6]);
```

## API

### `concat_arrays!(a, b, …)`

Macro shorthand. Zero arguments returns `[]`; one argument passes through
unchanged; two or more expand to `ArrayConcat::new([], first).append(second)…`.
Trailing commas are accepted.

### `ArrayConcat::new(a, b)`

Constructs `[…a…, …b…]`.

### `.append(c)`

Produces `[…self…, …c…]`.

### `.prepend(c)`

Produces `[…c…, …self…]`.

### `.insert(c)`

Produces `[…A…, …c…, …B…]` — inserts `c` between the `A` and `B` halves of
`self`.

### `.into_array::<N>()`

Moves the concatenation into a `[T; N]`. Compile-time error if `N ≠ SIZE`.

### `.as_ref()` / `.as_mut()`

Borrows all elements as `&[T]` / `&mut [T]`.

### `ConcatableArray` trait

The unsafe marker trait implemented by `[T; N]` and `ArrayConcat`. The
implementing type must have no padding — its in-memory representation must be
exactly `SIZE * size_of::<T>()` bytes.

## `no_std`

The crate is `#![no_std]`. The only dependency is
[`const_panic`](https://crates.io/crates/const_panic), used for the
compile-time size assertion in `into_array`.
