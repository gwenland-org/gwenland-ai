//! `#[repr(C)]` mirrors of the PJRT C API structs and enums.
//!
//! Field order and field types here are ABI, not style. See [`super`] for the
//! audit contract. C `enum` is `int` on every platform PJRT ships for, hence
//! `#[repr(i32)]`.
#![allow(non_snake_case)] // field names mirror the C header 1:1, deliberately.

use core::ffi::{c_char, c_int, c_void};

// ---------------------------------------------------------------------------
// Opaque handles — never constructed or dereferenced by gljax.
// ---------------------------------------------------------------------------

macro_rules! opaque {
    ($($name:ident),* $(,)?) => {$(
        /// Opaque PJRT handle. Only ever held behind a pointer.
        #[repr(C)]
        pub struct $name {
            _data: [u8; 0],
            _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
        }
    )*};
}

opaque!(
    PjrtError,
    PjrtEvent,
    PjrtClient,
    PjrtDevice,
    PjrtMemory,
    PjrtBuffer,
    PjrtBufferMemoryLayout,
    PjrtExecutable,
    PjrtLoadedExecutable,
    PjrtExecuteContext,
    PjrtDeviceDescription,
    PjrtTopologyDescription,
);

// ---------------------------------------------------------------------------
// Extensions + version
// ---------------------------------------------------------------------------

/// `PJRT_Extension_Base`. gljax never sends extensions; `extension_start` is
/// always null. Declared so the pointer fields have a real type.
#[repr(C)]
pub struct PjrtExtensionBase {
    pub struct_size: usize,
    pub type_: i32,
    pub next: *mut PjrtExtensionBase,
}

/// `PJRT_Api_Version`. Embedded by value in [`PjrtApi`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PjrtApiVersion {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub major_version: c_int,
    pub minor_version: c_int,
}

/// `PJRT_API_MAJOR` in the header gljax was bound against.
///
/// A major bump means "a method or argument was deleted, an argument type
/// changed, or fields were rearranged" — none of which `struct_size` can
/// detect. Mismatch is refused outright (P5).
pub const PJRT_API_MAJOR: c_int = 0;

/// `PJRT_API_MINOR` in the header gljax was bound against.
///
/// Recorded for diagnostics only. A plugin with a *lower* minor is fine as
/// long as it carries the slots gljax calls, which is what the `struct_size`
/// check in `pjrt::plugin` actually tests. ARTX01 §9.1 recommends refusing on
/// any minor divergence; that would reject every plugin that is not this exact
/// build, so gljax checks reachability of the called slots instead.
pub const PJRT_API_MINOR: c_int = 114;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// `PJRT_Error_Code` — abseil status codes.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PjrtErrorCode {
    Ok = 0,
    Cancelled = 1,
    Unknown = 2,
    InvalidArgument = 3,
    DeadlineExceeded = 4,
    NotFound = 5,
    AlreadyExists = 6,
    PermissionDenied = 7,
    ResourceExhausted = 8,
    FailedPrecondition = 9,
    Aborted = 10,
    OutOfRange = 11,
    Unimplemented = 12,
    Internal = 13,
    Unavailable = 14,
    DataLoss = 15,
    Unauthenticated = 16,
}

/// `PJRT_Buffer_Type` — the on-device element type.
///
/// Discriminants are positional in the C enum, so the order below **is** the
/// ABI. Only the types gljax can currently transfer are named; the rest are
/// listed to keep the discriminants correct.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PjrtBufferType {
    Invalid = 0,
    Pred = 1,
    S8 = 2,
    S16 = 3,
    S32 = 4,
    S64 = 5,
    U8 = 6,
    U16 = 7,
    U32 = 8,
    U64 = 9,
    F16 = 10,
    F32 = 11,
    F64 = 12,
    Bf16 = 13,
    C64 = 14,
    C128 = 15,
    F8E5M2 = 16,
    F8E4M3Fn = 17,
    F8E4M3B11Fnuz = 18,
    F8E5M2Fnuz = 19,
    F8E4M3Fnuz = 20,
    S4 = 21,
    U4 = 22,
    Token = 23,
    S2 = 24,
    U2 = 25,
    F8E4M3 = 26,
    F8E3M4 = 27,
    F8E8M0Fnu = 28,
    F4E2M1Fn = 29,
    S1 = 30,
    U1 = 31,
    F6E2M3Fn = 32,
    F6E3M2Fn = 33,
}

/// `PJRT_HostBufferSemantics` — what the runtime may assume about the host
/// pointer handed to `PJRT_Client_BufferFromHostBuffer`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PjrtHostBufferSemantics {
    /// The runtime must be done with `data` when the call returns. This is the
    /// only semantics gljax uses: it means the caller's slice does not have to
    /// outlive the call, so no lifetime escapes into the buffer handle.
    ImmutableOnlyDuringCall = 0,
    ImmutableUntilTransferCompletes = 1,
    ImmutableZeroCopy = 2,
    MutableZeroCopy = 3,
}

// ---------------------------------------------------------------------------
// Args structs — every one starts with `struct_size` + `extension_start`.
// ---------------------------------------------------------------------------

/// `PJRT_Error_Destroy_Args`
#[repr(C)]
pub struct PjrtErrorDestroyArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub error: *mut PjrtError,
}

/// `PJRT_Error_Message_Args`
#[repr(C)]
pub struct PjrtErrorMessageArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub error: *const PjrtError,
    pub message: *const c_char,
    pub message_size: usize,
}

/// `PJRT_Error_GetCode_Args`
#[repr(C)]
pub struct PjrtErrorGetCodeArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub error: *const PjrtError,
    pub code: PjrtErrorCode,
}

/// `PJRT_Plugin_Initialize_Args`
#[repr(C)]
pub struct PjrtPluginInitializeArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
}

/// `PJRT_Event_Destroy_Args`
#[repr(C)]
pub struct PjrtEventDestroyArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub event: *mut PjrtEvent,
}

/// `PJRT_Event_Await_Args`
#[repr(C)]
pub struct PjrtEventAwaitArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub event: *mut PjrtEvent,
}

/// `PJRT_NamedValue`. gljax passes zero create-options, so this exists only to
/// give `create_options` a type.
#[repr(C)]
pub struct PjrtNamedValue {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub name: *const c_char,
    pub name_size: usize,
    pub type_: i32,
    /// The C type is a union of `{const char*, int64, const int64*, float, bool}`;
    /// every variant is at most pointer-sized, so an opaque pointer-sized cell
    /// has the right size and alignment.
    pub value: *const c_void,
    pub value_size: usize,
}

/// `PJRT_Client_Create_Args`
#[repr(C)]
pub struct PjrtClientCreateArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub create_options: *const PjrtNamedValue,
    pub num_options: usize,
    pub kv_get_callback: *mut c_void,
    pub kv_get_user_arg: *mut c_void,
    pub kv_put_callback: *mut c_void,
    pub kv_put_user_arg: *mut c_void,
    pub client: *mut PjrtClient,
    pub kv_try_get_callback: *mut c_void,
    pub kv_try_get_user_arg: *mut c_void,
}

/// `PJRT_Client_Destroy_Args`
#[repr(C)]
pub struct PjrtClientDestroyArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub client: *mut PjrtClient,
}

/// `PJRT_Client_PlatformName_Args`
#[repr(C)]
pub struct PjrtClientPlatformNameArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub client: *mut PjrtClient,
    pub platform_name: *const c_char,
    pub platform_name_size: usize,
}

/// `PJRT_Client_PlatformVersion_Args`
#[repr(C)]
pub struct PjrtClientPlatformVersionArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub client: *mut PjrtClient,
    pub platform_version: *const c_char,
    pub platform_version_size: usize,
}

/// `PJRT_Client_AddressableDevices_Args`
#[repr(C)]
pub struct PjrtClientAddressableDevicesArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub client: *mut PjrtClient,
    pub addressable_devices: *const *mut PjrtDevice,
    pub num_addressable_devices: usize,
}

/// `PJRT_Program` — the compile input. `format` selects how `code` is read;
/// gljax always sends `"mlir"` with StableHLO **text**.
#[repr(C)]
pub struct PjrtProgram {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub code: *mut c_char,
    pub code_size: usize,
    pub format: *const c_char,
    pub format_size: usize,
}

/// `PJRT_Client_Compile_Args`
#[repr(C)]
pub struct PjrtClientCompileArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub client: *mut PjrtClient,
    pub program: *const PjrtProgram,
    /// Serialized `CompileOptionsProto`. See `pjrt::compile::compile_options`.
    pub compile_options: *const c_char,
    pub compile_options_size: usize,
    pub executable: *mut PjrtLoadedExecutable,
}

/// `PJRT_Client_BufferFromHostBuffer_Args`
#[repr(C)]
pub struct PjrtClientBufferFromHostBufferArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub client: *mut PjrtClient,
    pub data: *const c_void,
    pub type_: PjrtBufferType,
    pub dims: *const i64,
    pub num_dims: usize,
    pub byte_strides: *const i64,
    pub num_byte_strides: usize,
    pub host_buffer_semantics: PjrtHostBufferSemantics,
    pub device: *mut PjrtDevice,
    pub memory: *mut PjrtMemory,
    pub device_layout: *mut PjrtBufferMemoryLayout,
    pub done_with_host_buffer: *mut PjrtEvent,
    pub buffer: *mut PjrtBuffer,
}

/// `PJRT_Executable_Destroy_Args`
#[repr(C)]
pub struct PjrtExecutableDestroyArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub executable: *mut PjrtExecutable,
}

/// `PJRT_Executable_NumOutputs_Args`
#[repr(C)]
pub struct PjrtExecutableNumOutputsArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub executable: *mut PjrtExecutable,
    pub num_outputs: usize,
}

/// `PJRT_LoadedExecutable_Destroy_Args`
#[repr(C)]
pub struct PjrtLoadedExecutableDestroyArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub executable: *mut PjrtLoadedExecutable,
}

/// `PJRT_LoadedExecutable_GetExecutable_Args`
#[repr(C)]
pub struct PjrtLoadedExecutableGetExecutableArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub loaded_executable: *mut PjrtLoadedExecutable,
    pub executable: *mut PjrtExecutable,
}

/// `PJRT_ExecuteOptions`
///
/// ⚠️ Every field after `launch_id` is left zeroed by gljax, which is the
/// documented "no send/recv callbacks, no donation overrides, no multi-slice"
/// configuration. `struct_size` still reports the full struct so the plugin
/// knows the trailing fields were deliberately zeroed rather than absent.
#[repr(C)]
pub struct PjrtExecuteOptions {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub send_callbacks: *mut *mut c_void,
    pub recv_callbacks: *mut *mut c_void,
    pub num_send_ops: usize,
    pub num_recv_ops: usize,
    pub launch_id: c_int,
    pub non_donatable_input_indices: *const i64,
    pub num_non_donatable_input_indices: usize,
    pub context: *mut PjrtExecuteContext,
    pub call_location: *const c_char,
    pub num_tasks: usize,
    pub task_ids: *mut c_int,
    pub incarnation_ids: *mut i64,
    pub multi_slice_config: *mut c_void,
    pub use_major_to_minor_data_layout_for_callbacks: bool,
    pub hlo_output_callbacks: *mut c_void,
    pub num_hlo_output_callbacks: usize,
}

/// `PJRT_LoadedExecutable_Execute_Args`
#[repr(C)]
pub struct PjrtLoadedExecutableExecuteArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub executable: *mut PjrtLoadedExecutable,
    pub options: *mut PjrtExecuteOptions,
    pub argument_lists: *const *const *mut PjrtBuffer,
    pub num_devices: usize,
    pub num_args: usize,
    pub output_lists: *const *mut *mut PjrtBuffer,
    pub device_complete_events: *mut *mut PjrtEvent,
    pub execute_device: *mut PjrtDevice,
}

/// `PJRT_Buffer_Destroy_Args`
#[repr(C)]
pub struct PjrtBufferDestroyArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub buffer: *mut PjrtBuffer,
}

/// `PJRT_Buffer_ElementType_Args`
#[repr(C)]
pub struct PjrtBufferElementTypeArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub buffer: *mut PjrtBuffer,
    pub type_: PjrtBufferType,
}

/// `PJRT_Buffer_Dimensions_Args`
#[repr(C)]
pub struct PjrtBufferDimensionsArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub buffer: *mut PjrtBuffer,
    pub dims: *const i64,
    pub num_dims: usize,
}

/// `PJRT_Buffer_ToHostBuffer_Args`
#[repr(C)]
pub struct PjrtBufferToHostBufferArgs {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub src: *mut PjrtBuffer,
    pub host_layout: *mut PjrtBufferMemoryLayout,
    pub dst: *mut c_void,
    pub dst_size: usize,
    pub event: *mut PjrtEvent,
}

// ---------------------------------------------------------------------------
// The vtable
// ---------------------------------------------------------------------------

/// `PJRT_Api` — the plugin's function table, obtained from `GetPjrtApi()`.
///
/// ⛔ **Field order is ABI.** 138 function-pointer slots in header order. Slots
/// gljax calls are typed; the rest are `*mut c_void` placeholders that occupy
/// exactly one pointer each and cannot be invoked.
///
/// New slots are only ever appended by XLA, so a plugin built against an older
/// header simply has a smaller `struct_size`. `pjrt::plugin` checks that
/// `struct_size` reaches past the last slot gljax actually calls.
#[repr(C)]
pub struct PjrtApi {
    pub struct_size: usize,
    pub extension_start: *mut PjrtExtensionBase,
    pub pjrt_api_version: PjrtApiVersion,

    // 1..=5
    pub PJRT_Error_Destroy: Option<ffi::PjrtErrorDestroyFn>,
    pub PJRT_Error_Message: Option<ffi::PjrtErrorMessageFn>,
    pub PJRT_Error_GetCode: Option<ffi::PjrtErrorGetCodeFn>,
    pub PJRT_Plugin_Initialize: Option<ffi::PjrtPluginInitializeFn>,
    pub PJRT_Plugin_Attributes: *mut c_void,

    // 6..=10
    pub PJRT_Event_Destroy: Option<ffi::PjrtEventDestroyFn>,
    pub PJRT_Event_IsReady: *mut c_void,
    pub PJRT_Event_Error: *mut c_void,
    pub PJRT_Event_Await: Option<ffi::PjrtEventAwaitFn>,
    pub PJRT_Event_OnReady: *mut c_void,

    // 11..=23
    pub PJRT_Client_Create: Option<ffi::PjrtClientCreateFn>,
    pub PJRT_Client_Destroy: Option<ffi::PjrtClientDestroyFn>,
    pub PJRT_Client_PlatformName: Option<ffi::PjrtClientPlatformNameFn>,
    pub PJRT_Client_ProcessIndex: *mut c_void,
    pub PJRT_Client_PlatformVersion: Option<ffi::PjrtClientPlatformVersionFn>,
    pub PJRT_Client_Devices: *mut c_void,
    pub PJRT_Client_AddressableDevices: Option<ffi::PjrtClientAddressableDevicesFn>,
    pub PJRT_Client_LookupDevice: *mut c_void,
    pub PJRT_Client_LookupAddressableDevice: *mut c_void,
    pub PJRT_Client_AddressableMemories: *mut c_void,
    pub PJRT_Client_Compile: Option<ffi::PjrtClientCompileFn>,
    pub PJRT_Client_DefaultDeviceAssignment: *mut c_void,
    pub PJRT_Client_BufferFromHostBuffer: Option<ffi::PjrtClientBufferFromHostBufferFn>,

    // 24..=29
    pub PJRT_DeviceDescription_Id: *mut c_void,
    pub PJRT_DeviceDescription_ProcessIndex: *mut c_void,
    pub PJRT_DeviceDescription_Attributes: *mut c_void,
    pub PJRT_DeviceDescription_Kind: *mut c_void,
    pub PJRT_DeviceDescription_DebugString: *mut c_void,
    pub PJRT_DeviceDescription_ToString: *mut c_void,

    // 30..=35
    pub PJRT_Device_GetDescription: *mut c_void,
    pub PJRT_Device_IsAddressable: *mut c_void,
    pub PJRT_Device_LocalHardwareId: *mut c_void,
    pub PJRT_Device_AddressableMemories: *mut c_void,
    pub PJRT_Device_DefaultMemory: *mut c_void,
    pub PJRT_Device_MemoryStats: *mut c_void,

    // 36..=40
    pub PJRT_Memory_Id: *mut c_void,
    pub PJRT_Memory_Kind: *mut c_void,
    pub PJRT_Memory_DebugString: *mut c_void,
    pub PJRT_Memory_ToString: *mut c_void,
    pub PJRT_Memory_AddressableByDevices: *mut c_void,

    // 41..=50
    pub PJRT_Executable_Destroy: Option<ffi::PjrtExecutableDestroyFn>,
    pub PJRT_Executable_Name: *mut c_void,
    pub PJRT_Executable_NumReplicas: *mut c_void,
    pub PJRT_Executable_NumPartitions: *mut c_void,
    pub PJRT_Executable_NumOutputs: Option<ffi::PjrtExecutableNumOutputsFn>,
    pub PJRT_Executable_SizeOfGeneratedCodeInBytes: *mut c_void,
    pub PJRT_Executable_GetCostAnalysis: *mut c_void,
    pub PJRT_Executable_OutputMemoryKinds: *mut c_void,
    pub PJRT_Executable_OptimizedProgram: *mut c_void,
    pub PJRT_Executable_Serialize: *mut c_void,

    // 51..=58
    pub PJRT_LoadedExecutable_Destroy: Option<ffi::PjrtLoadedExecutableDestroyFn>,
    pub PJRT_LoadedExecutable_GetExecutable: Option<ffi::PjrtLoadedExecutableGetExecutableFn>,
    pub PJRT_LoadedExecutable_AddressableDevices: *mut c_void,
    pub PJRT_LoadedExecutable_Delete: *mut c_void,
    pub PJRT_LoadedExecutable_IsDeleted: *mut c_void,
    pub PJRT_LoadedExecutable_Execute: Option<ffi::PjrtLoadedExecutableExecuteFn>,
    pub PJRT_Executable_DeserializeAndLoad: *mut c_void,
    pub PJRT_LoadedExecutable_Fingerprint: *mut c_void,

    // 59..=77
    pub PJRT_Buffer_Destroy: Option<ffi::PjrtBufferDestroyFn>,
    pub PJRT_Buffer_ElementType: Option<ffi::PjrtBufferElementTypeFn>,
    pub PJRT_Buffer_Dimensions: Option<ffi::PjrtBufferDimensionsFn>,
    pub PJRT_Buffer_UnpaddedDimensions: *mut c_void,
    pub PJRT_Buffer_DynamicDimensionIndices: *mut c_void,
    pub PJRT_Buffer_GetMemoryLayout: *mut c_void,
    pub PJRT_Buffer_OnDeviceSizeInBytes: *mut c_void,
    pub PJRT_Buffer_Device: *mut c_void,
    pub PJRT_Buffer_Memory: *mut c_void,
    pub PJRT_Buffer_Delete: *mut c_void,
    pub PJRT_Buffer_IsDeleted: *mut c_void,
    pub PJRT_Buffer_CopyToDevice: *mut c_void,
    pub PJRT_Buffer_ToHostBuffer: Option<ffi::PjrtBufferToHostBufferFn>,
    pub PJRT_Buffer_IsOnCpu: *mut c_void,
    pub PJRT_Buffer_ReadyEvent: *mut c_void,
    pub PJRT_Buffer_UnsafePointer: *mut c_void,
    pub PJRT_Buffer_IncreaseExternalReferenceCount: *mut c_void,
    pub PJRT_Buffer_DecreaseExternalReferenceCount: *mut c_void,
    pub PJRT_Buffer_OpaqueDeviceMemoryDataPointer: *mut c_void,

    // 78..=82
    pub PJRT_CopyToDeviceStream_Destroy: *mut c_void,
    pub PJRT_CopyToDeviceStream_AddChunk: *mut c_void,
    pub PJRT_CopyToDeviceStream_TotalBytes: *mut c_void,
    pub PJRT_CopyToDeviceStream_GranuleSize: *mut c_void,
    pub PJRT_CopyToDeviceStream_CurrentBytes: *mut c_void,

    // 83..=89
    pub PJRT_TopologyDescription_Create: *mut c_void,
    pub PJRT_TopologyDescription_Destroy: *mut c_void,
    pub PJRT_TopologyDescription_PlatformName: *mut c_void,
    pub PJRT_TopologyDescription_PlatformVersion: *mut c_void,
    pub PJRT_TopologyDescription_GetDeviceDescriptions: *mut c_void,
    pub PJRT_TopologyDescription_Serialize: *mut c_void,
    pub PJRT_TopologyDescription_Attributes: *mut c_void,

    // 90
    pub PJRT_Compile: *mut c_void,

    // 91..=138 — appended after the last major bump; header order preserved.
    pub PJRT_Executable_OutputElementTypes: *mut c_void,
    pub PJRT_Executable_OutputDimensions: *mut c_void,
    pub PJRT_Buffer_CopyToMemory: *mut c_void,
    pub PJRT_Client_CreateViewOfDeviceBuffer: *mut c_void,
    pub PJRT_Executable_Fingerprint: *mut c_void,
    pub PJRT_Client_TopologyDescription: *mut c_void,
    pub PJRT_Executable_GetCompiledMemoryStats: *mut c_void,
    pub PJRT_Memory_Kind_Id: *mut c_void,
    pub PJRT_ExecuteContext_Create: *mut c_void,
    pub PJRT_ExecuteContext_Destroy: *mut c_void,
    pub PJRT_Buffer_CopyRawToHost: *mut c_void,
    pub PJRT_AsyncHostToDeviceTransferManager_Destroy: *mut c_void,
    pub PJRT_AsyncHostToDeviceTransferManager_TransferData: *mut c_void,
    pub PJRT_Client_CreateBuffersForAsyncHostToDevice: *mut c_void,
    pub PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer: *mut c_void,
    pub PJRT_AsyncHostToDeviceTransferManager_Device: *mut c_void,
    pub PJRT_AsyncHostToDeviceTransferManager_BufferCount: *mut c_void,
    pub PJRT_AsyncHostToDeviceTransferManager_BufferSize: *mut c_void,
    pub PJRT_AsyncHostToDeviceTransferManager_SetBufferError: *mut c_void,
    pub PJRT_AsyncHostToDeviceTransferManager_AddMetadata: *mut c_void,
    pub PJRT_Client_DmaMap: *mut c_void,
    pub PJRT_Client_DmaUnmap: *mut c_void,
    pub PJRT_Client_CreateUninitializedBuffer: *mut c_void,
    pub PJRT_Client_UpdateGlobalProcessInfo: *mut c_void,
    pub PJRT_TopologyDescription_Deserialize: *mut c_void,
    pub PJRT_Client_CreateAliasBuffer: *mut c_void,
    pub PJRT_Client_FulfillAliasBuffer: *mut c_void,
    pub PJRT_LoadedExecutable_GetDeviceAssignment: *mut c_void,
    pub PJRT_Client_CreateErrorBuffer: *mut c_void,
    pub PJRT_AsyncHostToDeviceTransferManager_TransferLiteral: *mut c_void,
    pub PJRT_Buffer_CopyRawToHostFuture: *mut c_void,
    pub PJRT_Device_PoisonExecution: *mut c_void,
    pub PJRT_Device_CreateAsyncTrackingEvent: *mut c_void,
    pub PJRT_AsyncTrackingEvent_Destroy: *mut c_void,
    pub PJRT_Executable_GetCompileOptions: *mut c_void,
    pub PJRT_Buffer_DonateWithControlDependency: *mut c_void,
    pub PJRT_Event_Create: *mut c_void,
    pub PJRT_Event_Set: *mut c_void,
    pub PJRT_Device_GetAttributes: *mut c_void,
    pub PJRT_Client_Load: *mut c_void,
    pub PJRT_LoadedExecutable_AddressableDeviceLogicalIds: *mut c_void,
    pub PJRT_Buffer_Bitcast: *mut c_void,
    pub PJRT_Error_ForEachPayload: *mut c_void,
    pub PJRT_TopologyDescription_Fingerprint: *mut c_void,
    pub PJRT_Executable_ParameterMemoryKinds: *mut c_void,
    pub PJRT_Device_ClearMemoryStats: *mut c_void,
    pub PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace: *mut c_void,
    pub PJRT_TopologyDescription_GetMemorySpaceKindIds: *mut c_void,
}

use super::ffi;

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, offset_of, size_of};

    /// The vtable's shape is the whole ABI contract. If this drifts, every
    /// call gljax makes lands on the wrong function — silently, because a
    /// function pointer is a function pointer.
    ///
    /// `PJRT_Api` = struct_size + extension_start + PjrtApiVersion + 138 slots.
    /// `PjrtApiVersion` is size_t + ptr + int + int = 24 bytes on LP64/LLP64.
    #[test]
    fn pjrt_api_vtable_has_exactly_138_slots() {
        const HEADER_SLOT_COUNT: usize = 138;
        let ptr = size_of::<*mut c_void>();
        let prefix = size_of::<usize>() + ptr + size_of::<PjrtApiVersion>();
        assert_eq!(
            size_of::<PjrtApi>(),
            prefix + HEADER_SLOT_COUNT * ptr,
            "PjrtApi slot count drifted from pjrt_c_api.h (PJRT_API_MINOR {PJRT_API_MINOR})"
        );
    }

    #[test]
    fn pjrt_api_version_is_24_bytes() {
        // size_t(8) + ptr(8) + int(4) + int(4). Embedded by value, so a wrong
        // size here shifts every function pointer that follows it.
        assert_eq!(size_of::<PjrtApiVersion>(), 24);
        assert_eq!(offset_of!(PjrtApiVersion, major_version), 16);
        assert_eq!(offset_of!(PjrtApiVersion, minor_version), 20);
    }

    /// Spot-check the slots gljax actually calls. These offsets are what the
    /// `struct_size` reachability check in `pjrt::plugin` is stated in terms of.
    #[test]
    fn called_slots_sit_at_their_header_indices() {
        let ptr = size_of::<*mut c_void>();
        let prefix = size_of::<usize>() + ptr + size_of::<PjrtApiVersion>();
        // (field offset, 1-based index in the header's slot list)
        let cases = [
            (offset_of!(PjrtApi, PJRT_Error_Destroy), 1),
            (offset_of!(PjrtApi, PJRT_Plugin_Initialize), 4),
            (offset_of!(PjrtApi, PJRT_Event_Await), 9),
            (offset_of!(PjrtApi, PJRT_Client_Create), 11),
            (offset_of!(PjrtApi, PJRT_Client_Compile), 21),
            (offset_of!(PjrtApi, PJRT_Client_BufferFromHostBuffer), 23),
            (offset_of!(PjrtApi, PJRT_Executable_NumOutputs), 45),
            (offset_of!(PjrtApi, PJRT_LoadedExecutable_Execute), 56),
            (offset_of!(PjrtApi, PJRT_Buffer_ToHostBuffer), 71),
            (
                offset_of!(PjrtApi, PJRT_TopologyDescription_GetMemorySpaceKindIds),
                138,
            ),
        ];
        for (offset, index) in cases {
            assert_eq!(
                offset,
                prefix + (index - 1) * ptr,
                "slot {index} moved — audit against pjrt_c_api.h"
            );
        }
    }

    /// Args structs are `size_t`-aligned and start with `struct_size`, so
    /// `struct_size!` on the last field must never exceed `size_of`.
    #[test]
    fn args_struct_sizes_match_the_c_macro() {
        assert_eq!(offset_of!(PjrtErrorMessageArgs, struct_size), 0);
        assert_eq!(
            crate::struct_size!(PjrtErrorMessageArgs, message_size: usize),
            size_of::<PjrtErrorMessageArgs>()
        );
        assert_eq!(
            crate::struct_size!(PjrtClientCreateArgs, kv_try_get_user_arg: *mut c_void),
            size_of::<PjrtClientCreateArgs>()
        );
        assert_eq!(
            crate::struct_size!(PjrtProgram, format_size: usize),
            size_of::<PjrtProgram>()
        );
        assert_eq!(
            crate::struct_size!(PjrtLoadedExecutableExecuteArgs, execute_device: *mut PjrtDevice),
            size_of::<PjrtLoadedExecutableExecuteArgs>()
        );
        // `PjrtExecuteOptions` ends in a `usize` after a `bool`, so the C
        // macro's value is the padded size here too.
        assert_eq!(
            crate::struct_size!(PjrtExecuteOptions, num_hlo_output_callbacks: usize),
            size_of::<PjrtExecuteOptions>()
        );
        assert_eq!(align_of::<PjrtExecuteOptions>(), align_of::<usize>());
    }

    #[test]
    fn buffer_type_discriminants_match_the_c_enum() {
        // Positional in C; F32/BF16 are the two gljax transfers today, and an
        // off-by-one here would silently reinterpret every byte transferred.
        assert_eq!(PjrtBufferType::F32 as i32, 11);
        assert_eq!(PjrtBufferType::Bf16 as i32, 13);
        assert_eq!(PjrtBufferType::F64 as i32, 12);
        assert_eq!(PjrtBufferType::S32 as i32, 4);
        assert_eq!(PjrtBufferType::F6E3M2Fn as i32, 33);
    }
}
