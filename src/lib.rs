#![feature(core_private_bignum, core_float, core_intrinsics, const_fn, associated_type_defaults)]
#![allow(unused_variables, unused_imports, dead_code, unused_macros)]


#![no_std]


use core::intrinsics;
use core::mem;
use core::fmt;
use core::cmp;
use core::num;

// use core::num::Float;
// use core::num::bignum;
use core::num::ParseIntError;
use core::clone::Clone;
use core::marker::Copy;
use core::option::Option;
use core::result::Result;

use core::ops::{
    Add, AddAssign,
    Sub, SubAssign,
    Mul, MulAssign,
    Div, DivAssign,
    Rem, RemAssign,
    Not, 
};


pub trait NumberTheory<RHS=Self> {
    type Output;

    fn is_zero(&self) -> bool;
    fn is_multiple_of(&self, other: &Self) -> bool;
    fn is_even(&self) -> bool;
    fn is_odd(&self) -> bool;

    fn gcd(&self, other: RHS) -> Self::Output;
    fn lcm(&self, other: &Self) -> Self;
    fn pow(&self, exponent: usize) -> Self;
    fn pow_mod(&self, exponent: &Self, modulo: &Self) -> Self;
}


/// Arithmetic operations required by bignums.
trait FullOps: Sized {
    /// Returns `(carry', v')` such that `carry' * 2^W + v' = self + other + carry`,
    /// where `W` is the number of bits in `Self`.
    fn full_add(self, other: Self, carry: bool) -> (bool /* carry */, Self);

    /// Returns `(carry', v')` such that `carry' * 2^W + v' = self * other + carry`,
    /// where `W` is the number of bits in `Self`.
    fn full_mul(self, other: Self, carry: Self) -> (Self /* carry */, Self);

    /// Returns `(carry', v')` such that `carry' * 2^W + v' = self * other + other2 + carry`,
    /// where `W` is the number of bits in `Self`.
    fn full_mul_add(self, other: Self, other2: Self, carry: Self) -> (Self /* carry */, Self);

    /// Returns `(quo, rem)` such that `borrow * 2^W + self = quo * other + rem`
    /// and `0 <= rem < other`, where `W` is the number of bits in `Self`.
    fn full_div_rem(self, other: Self, borrow: Self) -> (Self /* quotient */, Self /* remainder */);
}

macro_rules! impl_full_ops {
    ($($ty:ty: add($addfn:path), mul/div($bigty:ident);)*) => (
        $(
            impl FullOps for $ty {
                fn full_add(self, other: $ty, carry: bool) -> (bool, $ty) {
                    // this cannot overflow, the output is between 0 and 2*2^nbits - 1
                    // FIXME will LLVM optimize this into ADC or similar???
                    let (v, carry1) = unsafe { intrinsics::add_with_overflow(self, other) };
                    let (v, carry2) = unsafe {
                        intrinsics::add_with_overflow(v, if carry {1} else {0})
                    };
                    (carry1 || carry2, v)
                }

                fn full_mul(self, other: $ty, carry: $ty) -> ($ty, $ty) {
                    // this cannot overflow, the output is between 0 and 2^nbits * (2^nbits - 1)
                    let nbits = mem::size_of::<$ty>() * 8;
                    let v = (self as $bigty) * (other as $bigty) + (carry as $bigty);
                    ((v >> nbits) as $ty, v as $ty)
                }

                fn full_mul_add(self, other: $ty, other2: $ty, carry: $ty) -> ($ty, $ty) {
                    // this cannot overflow, the output is between 0 and 2^(2*nbits) - 1
                    let nbits = mem::size_of::<$ty>() * 8;
                    let v = (self as $bigty) * (other as $bigty) + (other2 as $bigty) +
                            (carry as $bigty);
                    ((v >> nbits) as $ty, v as $ty)
                }

                fn full_div_rem(self, other: $ty, borrow: $ty) -> ($ty, $ty) {
                    debug_assert!(borrow < other);
                    // this cannot overflow, the dividend is between 0 and other * 2^nbits - 1
                    let nbits = mem::size_of::<$ty>() * 8;
                    let lhs = ((borrow as $bigty) << nbits) | (self as $bigty);
                    let rhs = other as $bigty;
                    ((lhs / rhs) as $ty, (lhs % rhs) as $ty)
                }
            }
        )*
    )
}

impl_full_ops! {
    u8:  add(intrinsics::u8_add_with_overflow),  mul/div(u16);
    u16: add(intrinsics::u16_add_with_overflow), mul/div(u32);
    u32: add(intrinsics::u32_add_with_overflow), mul/div(u64);
//  u64: add(intrinsics::u64_add_with_overflow), mul/div(u128); // see RFC #521 for enabling this.
}


/// Table of powers of 5 representable in digits. Specifically, the largest {u8, u16, u32} value
/// that's a power of five, plus the corresponding exponent. Used in `mul_pow5`.
const SMALL_POW5: [(u64, usize); 3] = [(125, 3), (15625, 6), (1_220_703_125, 13)];

// https://github.com/rust-lang/rust/blob/master/src/libcore/num/bignum.rs
macro_rules! define_bignum {
    ($name:ident: type=$ty:ty, n=$n:expr) => (
        /// Stack-allocated arbitrary-precision (up to certain limit) integer.
        ///
        /// This is backed by a fixed-size array of given type ("digit").
        /// While the array is not very large (normally some hundred bytes),
        /// copying it recklessly may result in the performance hit.
        /// Thus this is intentionally not `Copy`.
        ///
        /// All operations available to bignums panic in the case of overflows.
        /// The caller is responsible to use large enough bignum types.
        #[allow(non_camel_case_types)]
        pub struct $name {
            /// One plus the offset to the maximum "digit" in use.
            /// This does not decrease, so be aware of the computation order.
            /// `base[size..]` should be zero.
            size: usize,
            /// Digits. `[a, b, c, ...]` represents `a + b*2^W + c*2^(2W) + ...`
            /// where `W` is the number of bits in the digit type.
            base: [$ty; $n]
        }

        impl $name {
            /// Makes a bignum from one digit.
            pub fn from_small(v: $ty) -> $name {
                let mut base = [0; $n];
                base[0] = v;
                $name { size: 1, base: base }
            }

            /// Makes a bignum from `u64` value.
            pub fn from_u64(mut v: u64) -> $name {
                use mem;

                let mut base = [0; $n];
                let mut sz = 0;
                while v > 0 {
                    base[sz] = v as $ty;
                    v >>= mem::size_of::<$ty>() * 8;
                    sz += 1;
                }
                $name { size: sz, base: base }
            }

            /// Returns the internal digits as a slice `[a, b, c, ...]` such that the numeric
            /// value is `a + b * 2^W + c * 2^(2W) + ...` where `W` is the number of bits in
            /// the digit type.
            pub fn digits(&self) -> &[$ty] {
                &self.base[..self.size]
            }

            /// Returns the `i`-th bit where bit 0 is the least significant one.
            /// In other words, the bit with weight `2^i`.
            pub fn get_bit(&self, i: usize) -> u8 {
                use mem;

                let digitbits = mem::size_of::<$ty>() * 8;
                let d = i / digitbits;
                let b = i % digitbits;
                ((self.base[d] >> b) & 1) as u8
            }

            /// Returns `true` if the bignum is zero.
            pub fn is_zero(&self) -> bool {
                self.digits().iter().all(|&v| v == 0)
            }

            /// Returns the number of bits necessary to represent this value. Note that zero
            /// is considered to need 0 bits.
            pub fn bit_length(&self) -> usize {
                // Skip over the most significant digits which are zero.
                let digits = self.digits();
                let zeros = digits.iter().rev().take_while(|&&x| x == 0).count();
                let end = digits.len() - zeros;
                let nonzero = &digits[..end];

                if nonzero.is_empty() {
                    // There are no non-zero digits, i.e. the number is zero.
                    return 0;
                }
                // This could be optimized with leading_zeros() and bit shifts, but that's
                // probably not worth the hassle.
                let digitbits = mem::size_of::<$ty>()* 8;
                let mut i = nonzero.len() * digitbits - 1;
                while self.get_bit(i) == 0 {
                    i -= 1;
                }
                i + 1
            }
        }


        impl_uint!($name, type=$ty, n=$n);

        impl_core_ops_add!($name, type=$ty, n=$n);
        impl_core_ops_sub!($name, type=$ty, n=$n);
        impl_core_ops_mul!($name, type=$ty, n=$n);
        impl_core_ops_rem!($name, type=$ty, n=$n);
        impl_core_ops_div!($name, type=$ty, n=$n);
        impl_core_ops_not!($name, type=$ty, n=$n);


        impl_core_cmp!($name, type=$ty, n=$n);
        impl_core_fmt!($name, type=$ty, n=$n);

        impl Copy for $name { }

        impl Clone for $name {
            fn clone(&self) -> $name {
                $name { size: self.size, base: self.base }
            }
        }

        impl_from_primitive! { i8, $name }
        impl_from_primitive! { u8, $name }
        impl_from_primitive! { i16, $name }
        impl_from_primitive! { u16, $name }
        impl_from_primitive! { i32, $name }
        impl_from_primitive! { u32, $name }
        impl_from_primitive! { i64, $name }
        impl_from_primitive! { u64, $name }
        impl_from_primitive! { isize, $name }
        impl_from_primitive! { usize, $name }
    )
}

macro_rules! impl_uint {
    ($name:ident, type=$ty:ty, n=$n:expr) => (
        impl $name {
            // https://github.com/rust-lang/rust/blob/master/src/libcore/num/mod.rs
            
            /// The smallest value that can be represented by this integer type.
            pub const MIN: $name = $name::min_value();
            /// The largest value that can be represented by this integer type.
            pub const MAX: $name = $name::max_value();

            
            #[inline]
            pub const fn min_value() -> $name {
                $name { size: 0, base: [0; $n] }
            }

            #[inline]
            pub const fn max_value() -> $name {
                $name { size: $n, base: [<$ty>::max_value(); $n] }
            }

            pub fn from_str_radix(src: &str, radix: u32) -> Result<Self, ParseIntError> {
                // from_str_radix(src, radix)
                unimplemented!()
            }

            #[inline]
            pub fn count_ones(self) -> u32 {
                unimplemented!()
            }

            #[inline]
            pub fn count_zeros(self) -> u32 {
                unimplemented!()
            }

            #[inline]
            pub fn leading_zeros(self) -> u32 {
                unimplemented!()
            }

            #[inline]
            pub fn trailing_zeros(self) -> u32 {
                unimplemented!()
            }

            pub fn rotate_left(self, n: u32) -> Self {
                unimplemented!()
            }

            pub fn rotate_right(self, n: u32) -> Self {
                unimplemented!()
            }

            pub fn swap_bytes(self) -> Self {
                unimplemented!()
            }

            pub fn reverse_bits(self) -> Self {
                unimplemented!()
            }

            pub fn from_be(x: Self) -> Self {
                unimplemented!()
            }

            pub fn from_le(x: Self) -> Self {
                unimplemented!()
            }

            pub fn to_be(self) -> Self {
                unimplemented!()
            }

            pub fn to_le(self) -> Self {
                unimplemented!()
            }
        }
    )
}

macro_rules! impl_core_ops_add {
    ($name:ident, type=$ty:ty, n=$n:expr) => (
        // /// Adds `other` to itself and returns its own mutable reference.
        // pub fn add<'a>(&'a mut self, other: &$name) -> &'a mut $name {
        //     let mut sz = cmp::max(self.size, other.size);
        //     let mut carry = false;
        //     for (a, b) in self.base[..sz].iter_mut().zip(&other.base[..sz]) {
        //         let (c, v) = (*a).full_add(*b, carry);
        //         *a = v;
        //         carry = c;
        //     }
        //     if carry {
        //         self.base[sz] = 1;
        //         sz += 1;
        //     }
        //     self.size = sz;
        //     self
        // }

        // pub fn add_small(&mut self, other: $ty) -> &mut $name {
        //     let (mut carry, v) = self.base[0].full_add(other, false);
        //     self.base[0] = v;
        //     let mut i = 1;
        //     while carry {
        //         let (c, v) = self.base[i].full_add(0, carry);
        //         self.base[i] = v;
        //         carry = c;
        //         i += 1;
        //     }
        //     if i > self.size {
        //         self.size = i;
        //     }
        //     self
        // }

        impl Add<$ty> for $name {
            type Output = $name;

            fn add(self, other: $ty) -> $name {
                let mut self_clone = self.clone();
                self_clone += other;
                self_clone
            }
        }

        impl AddAssign<$ty> for $name {
            fn add_assign(&mut self, other: $ty) {
                let (mut carry, v) = self.base[0].full_add(other, false);
                self.base[0] = v;
                let mut i = 1;
                while carry {
                    let (c, v) = self.base[i].full_add(0, carry);
                    self.base[i] = v;
                    carry = c;
                    i += 1;
                }
                if i > self.size {
                    self.size = i;
                }
            }
        }

        impl Add<$name> for $name {
            type Output = $name;

            fn add(self, other: $name) -> $name {
                let mut self_clone = self.clone();
                self_clone += other;
                self_clone
            }
        }

        impl AddAssign<$name> for $name {
            fn add_assign(&mut self, other: $name) {
                let mut sz = cmp::max(self.size, other.size);
                let mut carry = false;

                for (a, b) in self.base[..sz].iter_mut().zip(&other.base[..sz]) {
                    let (c, v) = (*a).full_add(*b, carry);
                    *a = v;
                    carry = c;
                }
                if carry {
                    self.base[sz] = 1;
                    sz += 1;
                }
                self.size = sz;
            }
        }


    )
}

macro_rules! impl_core_ops_sub {
    ($name:ident, type=$ty:ty, n=$n:expr) => (
        impl Sub<$name> for $name {
            type Output = $name;

            fn sub(self, other: $name) -> $name {
                let mut self_clone = self.clone();
                self_clone -= other.clone();
                self_clone
            }
        }

        impl SubAssign<$name> for $name {
            fn sub_assign(&mut self, other: $name) {
                let sz = cmp::max(self.size, other.size);
                let mut noborrow = true;
                for (a, b) in self.base[..sz].iter_mut().zip(&other.base[..sz]) {
                    let (c, v) = (*a).full_add(!*b, noborrow);
                    *a = v;
                    noborrow = c;
                }
                assert!(noborrow);
                self.size = sz;
            }
        }
        // /// Subtracts `other` from itself and returns its own mutable reference.
        // pub fn sub<'a>(&'a mut self, other: &$name) -> &'a mut $name {
        //     let sz = cmp::max(self.size, other.size);
        //     let mut noborrow = true;
        //     for (a, b) in self.base[..sz].iter_mut().zip(&other.base[..sz]) {
        //         let (c, v) = (*a).full_add(!*b, noborrow);
        //         *a = v;
        //         noborrow = c;
        //     }
        //     assert!(noborrow);
        //     self.size = sz;
        //     self
        // }
    )
}

macro_rules! impl_core_ops_mul {
    ($name:ident, type=$ty:ty, n=$n:expr) => (
        impl $name {
            /// Multiplies itself by a digit-sized `other` and returns its own
            /// mutable reference.
            fn mul_small(&mut self, other: $ty) -> &mut $name {
                let mut sz = self.size;
                let mut carry = 0;
                for a in &mut self.base[..sz] {
                    let (c, v) = (*a).full_mul(other, carry);
                    *a = v;
                    carry = c;
                }
                if carry > 0 {
                    self.base[sz] = carry;
                    sz += 1;
                }
                self.size = sz;
                self
            }

            /// Multiplies itself by `2^bits` and returns its own mutable reference.
            fn mul_pow2(&mut self, bits: usize) -> &mut $name {
                let digitbits = mem::size_of::<$ty>() * 8;
                let digits = bits / digitbits;
                let bits = bits % digitbits;

                assert!(digits < $n);
                debug_assert!(self.base[$n-digits..].iter().all(|&v| v == 0));
                debug_assert!(bits == 0 || (self.base[$n-digits-1] >> (digitbits - bits)) == 0);

                // shift by `digits * digitbits` bits
                for i in (0..self.size).rev() {
                    self.base[i+digits] = self.base[i];
                }
                for i in 0..digits {
                    self.base[i] = 0;
                }

                // shift by `bits` bits
                let mut sz = self.size + digits;
                if bits > 0 {
                    let last = sz;
                    let overflow = self.base[last-1] >> (digitbits - bits);
                    if overflow > 0 {
                        self.base[last] = overflow;
                        sz += 1;
                    }
                    for i in (digits+1..last).rev() {
                        self.base[i] = (self.base[i] << bits) |
                                       (self.base[i-1] >> (digitbits - bits));
                    }
                    self.base[digits] <<= bits;
                    // self.base[..digits] is zero, no need to shift
                }

                self.size = sz;
                self
            }

            /// Multiplies itself by `5^e` and returns its own mutable reference.
            fn mul_pow5(&mut self, mut e: usize) -> &mut $name {
                // There are exactly n trailing zeros on 2^n, and the only relevant digit sizes
                // are consecutive powers of two, so this is well suited index for the table.
                let table_index = mem::size_of::<$ty>().trailing_zeros() as usize;
                let (small_power, small_e) = SMALL_POW5[table_index];
                let small_power = small_power as $ty;

                // Multiply with the largest single-digit power as long as possible ...
                while e >= small_e {
                    self.mul_small(small_power);
                    e -= small_e;
                }

                // ... then finish off the remainder.
                let mut rest_power = 1;
                for _ in 0..e {
                    rest_power *= 5;
                }
                self.mul_small(rest_power);

                self
            }


            /// Multiplies itself by a number described by `other[0] + other[1] * 2^W +
            /// other[2] * 2^(2W) + ...` (where `W` is the number of bits in the digit type)
            /// and returns its own mutable reference.
            fn mul_digits<'a>(&'a mut self, other: &[$ty]) -> &'a mut $name {
                // the internal routine. works best when aa.len() <= bb.len().
                fn mul_inner(ret: &mut [$ty; $n], aa: &[$ty], bb: &[$ty]) -> usize {
                    let mut retsz = 0;
                    for (i, &a) in aa.iter().enumerate() {
                        if a == 0 { continue; }
                        let mut sz = bb.len();
                        let mut carry = 0;
                        for (j, &b) in bb.iter().enumerate() {
                            let (c, v) = a.full_mul_add(b, ret[i + j], carry);
                            ret[i + j] = v;
                            carry = c;
                        }
                        if carry > 0 {
                            ret[i + sz] = carry;
                            sz += 1;
                        }
                        if retsz < i + sz {
                            retsz = i + sz;
                        }
                    }
                    retsz
                }

                let mut ret = [0; $n];
                let retsz = if self.size < other.len() {
                    mul_inner(&mut ret, &self.digits(), other)
                } else {
                    mul_inner(&mut ret, other, &self.digits())
                };
                self.base = ret;
                self.size = retsz;
                self
            }
        }

        impl Mul<$ty> for $name {
            type Output = $name;

            fn mul(self, rhs: $ty) -> $name {
                let mut self_clone = self.clone();
                self_clone *= rhs;
                self_clone
            }
        }

        impl Mul<$name> for $name {
            type Output = $name;

            fn mul(self, rhs: $name) -> $name {
                let mut self_clone = self.clone();
                self_clone *= rhs;
                self_clone
            }
        }

        impl MulAssign<$ty> for $name {
            fn mul_assign(&mut self, other: $ty) {
                let mut sz = self.size;
                let mut carry = 0;
                for a in &mut self.base[..sz] {
                    let (c, v) = (*a).full_mul(other, carry);
                    *a = v;
                    carry = c;
                }
                if carry > 0 {
                    self.base[sz] = carry;
                    sz += 1;
                }
                self.size = sz;
            }
        }

        impl MulAssign<$name> for $name {
            fn mul_assign(&mut self, rhs: $name) {
                let other = &rhs.base[..];

                // the internal routine. works best when aa.len() <= bb.len().
                fn mul_inner(ret: &mut [$ty; $n], aa: &[$ty], bb: &[$ty]) -> usize {
                    let mut retsz = 0;
                    for (i, &a) in aa.iter().enumerate() {
                        if a == 0 { continue; }
                        let mut sz = bb.len();
                        let mut carry = 0;
                        for (j, &b) in bb.iter().enumerate() {
                            let (c, v) = a.full_mul_add(b, ret[i + j], carry);
                            ret[i + j] = v;
                            carry = c;
                        }
                        if carry > 0 {
                            ret[i + sz] = carry;
                            sz += 1;
                        }
                        if retsz < i + sz {
                            retsz = i + sz;
                        }
                    }
                    retsz
                }

                let mut ret = [0; $n];
                let retsz = if self.size < other.len() {
                    mul_inner(&mut ret, &self.digits(), other)
                } else {
                    mul_inner(&mut ret, other, &self.digits())
                };
                self.base = ret;
                self.size = retsz;
            }
        }
    )
}

macro_rules! impl_core_ops_rem {
    ($name:ident, type=$ty:ty, n=$n:expr) => (
        impl $name {
            /// Divides itself by a digit-sized `other` and returns its own
            /// mutable reference *and* the remainder.
            fn div_rem_small(&mut self, other: $ty) -> (&mut $name, $ty) {
                assert!(other > 0);

                let sz = self.size;
                let mut borrow = 0;
                for a in self.base[..sz].iter_mut().rev() {
                    let (q, r) = (*a).full_div_rem(other, borrow);
                    *a = q;
                    borrow = r;
                }
                (self, borrow)
            }

            /// Divide self by another bignum, overwriting `q` with the quotient and `r` with the
            /// remainder.
            fn div_rem(&self, d: &$name, q: &mut $name, r: &mut $name) {
                // Stupid slow base-2 long division taken from
                // https://en.wikipedia.org/wiki/Division_algorithm
                // FIXME use a greater base ($ty) for the long division.
                assert!(!d.is_zero());
                let digitbits = mem::size_of::<$ty>() * 8;
                for digit in &mut q.base[..] {
                    *digit = 0;
                }
                for digit in &mut r.base[..] {
                    *digit = 0;
                }
                r.size = d.size;
                q.size = 1;
                let mut q_is_zero = true;
                let end = self.bit_length();
                for i in (0..end).rev() {
                    r.mul_pow2(1);
                    r.base[0] |= self.get_bit(i) as $ty;
                    if &*r >= d {
                        // r.sub(d);
                        *r -= *d;
                        // Set bit `i` of q to 1.
                        let digit_idx = i / digitbits;
                        let bit_idx = i % digitbits;
                        if q_is_zero {
                            q.size = digit_idx + 1;
                            q_is_zero = false;
                        }
                        q.base[digit_idx] |= 1 << bit_idx;
                    }
                }
                debug_assert!(q.base[q.size..].iter().all(|&d| d == 0));
                debug_assert!(r.base[r.size..].iter().all(|&d| d == 0));
            }
        }

        impl Rem<$ty> for $name {
            type Output = $ty;

            fn rem(self, other: $ty) -> $ty {
                let mut self_clone = self.clone();
                let (_, r) = self_clone.div_rem_small(other);

                r
            }
        }

        impl RemAssign<$ty> for $name {
            fn rem_assign(&mut self, other: $ty) {
                *self = $name::from(self.rem(other));
            }
        }


        impl Rem<$name> for $name {
            type Output = $name;

            fn rem(self, other: $name) -> $name {
                let self_clone = self.clone();
                let mut q = $name::from(0u8);
                let mut r = $name::from(0u8);
                &self_clone.div_rem(&other, &mut q, &mut r);

                r
            }
        }

        impl RemAssign<$name> for $name {
            fn rem_assign(&mut self, other: $name) {
                *self = (*self).rem(other);
            }
        }
    )
}

macro_rules! impl_core_ops_div {
    ($name:ident, type=$ty:ty, n=$n:expr) => (
        impl Div<$ty> for $name {
            type Output = $name;

            fn div(self, other: $ty) -> $name {
                self.div($name::from(other))
            }
        }

        impl DivAssign<$ty> for $name {
            fn div_assign(&mut self, other: $ty) {
                *self /= $name::from(other);
            }
        }


        impl Div<$name> for $name {
            type Output = $name;

            fn div(self, other: $name) -> $name {
                let self_clone = self.clone();
                let mut q = $name::from(0u8);
                let mut r = $name::from(0u8);
                &self_clone.div_rem(&other, &mut q, &mut r);

                q
            }
        }

        impl DivAssign<$name> for $name {
            fn div_assign(&mut self, other: $name) {
                *self = (*self).div(other);
            }
        }
    )
}

macro_rules! impl_core_ops_not {
    ($name:ident, type=$ty:ty, n=$n:expr) => (
        impl Not for $name {
            type Output = $name;

            fn not(self) -> $name {
                $name::max_value() - self
            }
        }
    )
}


macro_rules! impl_core_ops {
    ($name:ident, type=$ty:ty, n=$n:expr) => (
        impl $name {
            
            
        }
    )
}

macro_rules! impl_core_cmp {
    ($name:ident, type=$ty:ty, n=$n:expr) => (
        impl cmp::PartialEq for $name {
            fn eq(&self, other: &$name) -> bool { self.base[..] == other.base[..] }
        }

        impl cmp::Eq for $name { }

        impl cmp::PartialOrd for $name {
            fn partial_cmp(&self, other: &$name) -> Option<cmp::Ordering> {
                Option::Some(self.cmp(other))
            }
        }

        impl cmp::Ord for $name {
            fn cmp(&self, other: &$name) -> cmp::Ordering {
                use cmp::max;
                let sz = max(self.size, other.size);
                let lhs = self.base[..sz].iter().cloned().rev();
                let rhs = other.base[..sz].iter().cloned().rev();
                lhs.cmp(rhs)
            }
        }
    )
}

macro_rules! impl_core_fmt {
    ($name:ident, type=$ty:ty, n=$n:expr) => (
        // impl fmt::Debug for $name {
        //     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        //         use mem;

        //         let sz = if self.size < 1 {1} else {self.size};
        //         let digitlen = mem::size_of::<$ty>() * 2;

        //         write!(f, "{:#x}", self.base[sz-1])?;
        //         for &v in self.base[..sz-1].iter().rev() {
        //             write!(f, "_{:01$x}", v, digitlen)?;
        //         }
        //         Result::Ok(())
        //     }
        // }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{:x}", self)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let sz = if self.size < 1 { 1 } else { self.size };
                let digitlen = mem::size_of::<$ty>() * 2;

                write!(f, "u{} {{ size: {}, base: {:?} }} ", $n*32, sz, &self.base[..sz] )
            }
        }

        impl fmt::LowerHex for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let sz = if self.size < 1 { 1 } else { self.size };
                let digitlen = mem::size_of::<$ty>() * 2;

                write!(f, "{:#0width$x}", self.base[sz-1], width=digitlen)?;
                for &v in self.base[..sz-1].iter().rev() {
                    write!(f, "_{:0width$x}", v, width=digitlen)?;
                }
                Result::Ok(())
            }
        }

        impl fmt::UpperHex for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let sz = if self.size < 1 { 1 } else { self.size };
                let digitlen = mem::size_of::<$ty>() * 2;

                write!(f, "{:#0width$X}", self.base[sz-1], width=digitlen)?;
                for &v in self.base[..sz-1].iter().rev() {
                    write!(f, "_{:0width$X}", v, width=digitlen)?;
                }
                Result::Ok(())
            }
        }

        impl fmt::Binary for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let sz = if self.size < 1 { 1 } else { self.size };
                let digitlen = mem::size_of::<$ty>() * 8;
                
                write!(f, "{:#0width$b}", self.base[sz-1], width=digitlen)?;
                for &v in self.base[..sz-1].iter().rev() {
                    write!(f, "_{:0width$b}", v, width=digitlen)?;
                }
                Result::Ok(())
            }
        }
    )
}


// Conversion traits for primitive integer and float types
// Conversions T -> T are covered by a blanket impl and therefore excluded
// Some conversions from and to usize/isize are not implemented due to portability concerns
macro_rules! impl_from_primitive {
    ($Small:ty, $Large:tt) => {
        impl From<$Small> for $Large {
            #[inline]
            fn from(small: $Small) -> $Large {
                $Large::from_u64(small as u64)
            }
        }
    }
}

macro_rules! impl_from_bignum {
    ($Small:ty, $Large:tt) => {
        impl From<$Small> for $Large {
            #[inline]
            fn from(small: $Small) -> $Large {
                $Large { size: small.size, base: self.base }
            }
        }
    }
}
macro_rules! impl_from_str_radix {
    ($($t:ty)*) => {$(
        impl FromStr for $t {
            type Err = ParseIntError;
            fn from_str(src: &str) -> Result<Self, ParseIntError> {
                // from_str_radix(src, 10)
                unimplemented!()
            }
        }
    )*}
}
// impl_from_str_radix! { isize i8 i16 i32 i64 i128 usize u8 u16 u32 u64 u128 }


define_bignum!(u128: type=u32, n=4);
define_bignum!(u256: type=u32, n=8);
define_bignum!(u512: type=u32, n=16);
define_bignum!(u1024: type=u32, n=32);
define_bignum!(u2048: type=u32, n=64);
define_bignum!(u4096: type=u32, n=128);
define_bignum!(u8192: type=u32, n=256);
define_bignum!(u16384: type=u32, n=512);


#[test]
fn test_add() {
    let a = u128::from(1);
    let b = u128::from(2);
    let c = u128::from(3);
    assert_eq!(a + b, c);
}

#[test]
fn test_sub() {
    let a = u128::from(10);
    let b = u128::from(2);
    let c = u128::from(8);
    assert_eq!(a - b, c);
}

#[test]
fn test_mul() {
    let a = u128::from(10);
    let b = u128::from(2);
    let c = u128::from(20);
    assert_eq!(a * b, c);
}

#[test]
fn test_div_rem() {
    let a = u128::from(10);
    let b = u128::from(2);
    let c = u128::from(5);
    assert_eq!(a / b, c);
    assert_eq!(a % b, u128::from(0));
}