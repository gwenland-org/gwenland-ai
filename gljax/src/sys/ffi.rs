//! `extern "C"` signatures for the PJRT vtable slots gljax calls.
//!
//! Every PJRT entry point takes exactly one `*_Args` pointer. Most return an
//! owned `PJRT_Error*` (null on success); the two error accessors return
//! `void`, because an error handler that could itself fail would be unusable.

use super::types::*;

/// `typedef void PJRT_Error_Destroy(PJRT_Error_Destroy_Args*)` — note: `void`.
pub type PjrtErrorDestroyFn = unsafe extern "C" fn(*mut PjrtErrorDestroyArgs);

/// `typedef void PJRT_Error_Message(PJRT_Error_Message_Args*)` — note: `void`.
pub type PjrtErrorMessageFn = unsafe extern "C" fn(*mut PjrtErrorMessageArgs);

pub type PjrtErrorGetCodeFn = unsafe extern "C" fn(*mut PjrtErrorGetCodeArgs) -> *mut PjrtError;

pub type PjrtPluginInitializeFn =
    unsafe extern "C" fn(*mut PjrtPluginInitializeArgs) -> *mut PjrtError;

pub type PjrtEventDestroyFn = unsafe extern "C" fn(*mut PjrtEventDestroyArgs) -> *mut PjrtError;
pub type PjrtEventAwaitFn = unsafe extern "C" fn(*mut PjrtEventAwaitArgs) -> *mut PjrtError;

pub type PjrtClientCreateFn = unsafe extern "C" fn(*mut PjrtClientCreateArgs) -> *mut PjrtError;
pub type PjrtClientDestroyFn = unsafe extern "C" fn(*mut PjrtClientDestroyArgs) -> *mut PjrtError;
pub type PjrtClientPlatformNameFn =
    unsafe extern "C" fn(*mut PjrtClientPlatformNameArgs) -> *mut PjrtError;
pub type PjrtClientPlatformVersionFn =
    unsafe extern "C" fn(*mut PjrtClientPlatformVersionArgs) -> *mut PjrtError;
pub type PjrtClientAddressableDevicesFn =
    unsafe extern "C" fn(*mut PjrtClientAddressableDevicesArgs) -> *mut PjrtError;
pub type PjrtClientCompileFn = unsafe extern "C" fn(*mut PjrtClientCompileArgs) -> *mut PjrtError;
pub type PjrtClientBufferFromHostBufferFn =
    unsafe extern "C" fn(*mut PjrtClientBufferFromHostBufferArgs) -> *mut PjrtError;

pub type PjrtExecutableDestroyFn =
    unsafe extern "C" fn(*mut PjrtExecutableDestroyArgs) -> *mut PjrtError;
pub type PjrtExecutableNumOutputsFn =
    unsafe extern "C" fn(*mut PjrtExecutableNumOutputsArgs) -> *mut PjrtError;

pub type PjrtLoadedExecutableDestroyFn =
    unsafe extern "C" fn(*mut PjrtLoadedExecutableDestroyArgs) -> *mut PjrtError;
pub type PjrtLoadedExecutableGetExecutableFn =
    unsafe extern "C" fn(*mut PjrtLoadedExecutableGetExecutableArgs) -> *mut PjrtError;
pub type PjrtLoadedExecutableExecuteFn =
    unsafe extern "C" fn(*mut PjrtLoadedExecutableExecuteArgs) -> *mut PjrtError;

pub type PjrtBufferDestroyFn = unsafe extern "C" fn(*mut PjrtBufferDestroyArgs) -> *mut PjrtError;
pub type PjrtBufferElementTypeFn =
    unsafe extern "C" fn(*mut PjrtBufferElementTypeArgs) -> *mut PjrtError;
pub type PjrtBufferDimensionsFn =
    unsafe extern "C" fn(*mut PjrtBufferDimensionsArgs) -> *mut PjrtError;
pub type PjrtBufferToHostBufferFn =
    unsafe extern "C" fn(*mut PjrtBufferToHostBufferArgs) -> *mut PjrtError;

/// The one symbol resolved from the plugin binary: `GetPjrtApi()`.
pub type GetPjrtApiFn = unsafe extern "C" fn() -> *const PjrtApi;
