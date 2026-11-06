//! Windows facts, asked of the OS directly.
//!
//! These used to come from spawning `wmic`, which Windows 11 removed. A missing
//! binary reads as "not observed", so on an up-to-date machine every probe here
//! went quiet at once and the report blamed the OS for lacking the counters.
//! The Win32 calls below need no subprocess and no crate, and each returns
//! `None`, never a guess, when the call fails.

use std::ffi::c_void;
use std::mem::{size_of, zeroed};

/// `ProcessorInformation` for `CallNtPowerInformation`.
const PROCESSOR_INFORMATION: i32 = 11;
/// `RelationProcessorCore`: one record per physical core.
const RELATION_PROCESSOR_CORE: i32 = 0;
/// `ALL_PROCESSOR_GROUPS`.
const ALL_PROCESSOR_GROUPS: u16 = 0xFFFF;

#[repr(C)]
struct FileTime {
    low: u32,
    high: u32,
}

impl FileTime {
    /// The value in seconds; a `FILETIME` duration counts 100 ns units.
    fn secs(&self) -> f64 {
        ((u64::from(self.high) << 32) | u64::from(self.low)) as f64 / 1e7
    }
}

#[repr(C)]
struct MemoryStatusEx {
    length: u32,
    memory_load: u32,
    total_phys: u64,
    avail_phys: u64,
    total_page_file: u64,
    avail_page_file: u64,
    total_virtual: u64,
    avail_virtual: u64,
    avail_extended_virtual: u64,
}

#[repr(C)]
struct ProcessMemoryCounters {
    cb: u32,
    page_fault_count: u32,
    peak_working_set_size: usize,
    working_set_size: usize,
    quota_peak_paged_pool_usage: usize,
    quota_paged_pool_usage: usize,
    quota_peak_non_paged_pool_usage: usize,
    quota_non_paged_pool_usage: usize,
    pagefile_usage: usize,
    peak_pagefile_usage: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcessorPowerInformation {
    number: u32,
    max_mhz: u32,
    current_mhz: u32,
    mhz_limit: u32,
    max_idle_state: u32,
    current_idle_state: u32,
}

#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcess() -> *mut c_void;
    fn GetProcessTimes(
        process: *mut c_void,
        creation: *mut FileTime,
        exit: *mut FileTime,
        kernel: *mut FileTime,
        user: *mut FileTime,
    ) -> i32;
    fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    fn K32GetProcessMemoryInfo(
        process: *mut c_void,
        counters: *mut ProcessMemoryCounters,
        size: u32,
    ) -> i32;
    fn GetLogicalProcessorInformationEx(
        relationship: i32,
        buffer: *mut c_void,
        returned_length: *mut u32,
    ) -> i32;
    fn GetActiveProcessorCount(group: u16) -> u32;
}

#[link(name = "powrprof")]
extern "system" {
    fn CallNtPowerInformation(
        level: i32,
        input: *mut c_void,
        input_len: u32,
        output: *mut c_void,
        output_len: u32,
    ) -> i32;
}

/// Kernel plus user CPU time of this process, in seconds.
pub(crate) fn process_cpu_secs() -> Option<f64> {
    // SAFETY: FileTime is plain integers, so all-zero is valid; the out
    // pointers are live locals and the pseudo-handle needs no closing.
    unsafe {
        let (mut creation, mut exit, mut kernel, mut user): (FileTime, FileTime, FileTime, FileTime) =
            (zeroed(), zeroed(), zeroed(), zeroed());
        let ok = GetProcessTimes(GetCurrentProcess(), &mut creation, &mut exit, &mut kernel, &mut user);
        (ok != 0).then(|| kernel.secs() + user.secs())
    }
}

/// This process's peak working set, in bytes.
pub(crate) fn peak_working_set_bytes() -> Option<u64> {
    // SAFETY: all-zero is valid for the counters struct, `cb` states its size
    // as the API requires, and the pointer is a live local.
    unsafe {
        let mut counters: ProcessMemoryCounters = zeroed();
        counters.cb = size_of::<ProcessMemoryCounters>() as u32;
        let ok = K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb);
        (ok != 0).then_some(counters.peak_working_set_size as u64)
    }
}

/// Total and available physical memory, in bytes.
pub(crate) fn physical_memory() -> Option<(u64, u64)> {
    // SAFETY: all-zero is valid for the struct, `length` is set as the API
    // requires, and the pointer is a live local.
    unsafe {
        let mut status: MemoryStatusEx = zeroed();
        status.length = size_of::<MemoryStatusEx>() as u32;
        (GlobalMemoryStatusEx(&mut status) != 0).then_some((status.total_phys, status.avail_phys))
    }
}

/// The mean current clock across logical processors, in MHz.
pub(crate) fn current_mhz() -> Option<f64> {
    // SAFETY: takes no pointers.
    let count = unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) } as usize;
    if count == 0 {
        return None;
    }
    let mut info = vec![ProcessorPowerInformation::default(); count];
    // SAFETY: the output buffer is live and its stated length is its true
    // size; a null input with zero length is what this level expects.
    let status = unsafe {
        CallNtPowerInformation(
            PROCESSOR_INFORMATION,
            std::ptr::null_mut(),
            0,
            info.as_mut_ptr().cast(),
            (count * size_of::<ProcessorPowerInformation>()) as u32,
        )
    };
    if status != 0 {
        return None;
    }
    let sum: u64 = info.iter().map(|p| u64::from(p.current_mhz)).sum();
    (sum > 0).then(|| sum as f64 / count as f64)
}

/// The number of physical cores.
pub(crate) fn physical_cores() -> Option<usize> {
    let mut len = 0u32;
    // SAFETY: a null buffer with length 0 is the documented way to ask for the
    // required size; the call fails with that size written to `len`.
    unsafe {
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len);
    }
    if len == 0 {
        return None;
    }
    // u64 elements so the records, which hold 8-byte fields, are aligned.
    let mut buffer = vec![0u64; (len as usize).div_ceil(8)];
    // SAFETY: the buffer holds at least `len` bytes and both pointers are live.
    let ok = unsafe {
        GetLogicalProcessorInformationEx(
            RELATION_PROCESSOR_CORE,
            buffer.as_mut_ptr().cast(),
            &mut len,
        )
    };
    if ok == 0 {
        return None;
    }
    count_records(&buffer, len as usize)
}

/// Walk variable-size records, each opening with `{ relationship: u32, size: u32 }`.
fn count_records(buffer: &[u64], len: usize) -> Option<usize> {
    let bytes: Vec<u8> = buffer.iter().flat_map(|word| word.to_ne_bytes()).collect();
    let (mut offset, mut records) = (0usize, 0usize);
    while offset + 8 <= len {
        let size = u32::from_ne_bytes(bytes[offset + 4..offset + 8].try_into().ok()?) as usize;
        if size == 0 {
            return None;
        }
        records += 1;
        offset += size;
    }
    (records > 0).then_some(records)
}

/// The CPU's own name, from the CPUID brand string.
#[cfg(target_arch = "x86_64")]
pub(crate) fn cpu_brand() -> Option<String> {
    use std::arch::x86_64::__cpuid;

    // SAFETY: CPUID exists on every x86_64 processor, and the extended leaves
    // are only read once leaf 0x80000000 reports them.
    #[allow(unused_unsafe)]
    let bytes: Vec<u8> = unsafe {
        if __cpuid(0x8000_0000).eax < 0x8000_0004 {
            return None;
        }
        (0x8000_0002u32..=0x8000_0004)
            .flat_map(|leaf| {
                let r = __cpuid(leaf);
                [r.eax, r.ebx, r.ecx, r.edx]
            })
            .flat_map(u32::to_le_bytes)
            .collect()
    };
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let brand = String::from_utf8_lossy(&bytes[..end]).trim().to_string();
    (!brand.is_empty()).then_some(brand)
}

/// No brand string outside x86_64.
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn cpu_brand() -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_probe_answers_on_this_machine() {
        assert!(cpu_brand().is_some_and(|b| b.len() > 3));
        let (total, avail) = physical_memory().expect("memory");
        assert!(total > 0 && avail <= total);
        assert!(physical_cores().is_some_and(|c| c >= 1));
        assert!(current_mhz().is_some_and(|m| m > 100.0));
    }

    #[test]
    fn cpu_time_advances_with_work() {
        let before = process_cpu_secs().expect("cpu time");
        let mut x = 0u64;
        while process_cpu_secs().expect("cpu time") <= before {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        }
        std::hint::black_box(x);
    }

    #[test]
    fn records_are_walked_by_their_stated_size() {
        // Two records of 16 and 8 bytes, then trailing bytes past `len`.
        let mut bytes = vec![0u8; 32];
        bytes[4] = 16;
        bytes[16 + 4] = 8;
        let words: Vec<u64> = bytes
            .chunks(8)
            .map(|c| u64::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(count_records(&words, 24), Some(2));
        assert_eq!(count_records(&[0u64; 4], 32), None, "a zero size must not loop");
    }
}
