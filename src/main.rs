#![windows_subsystem = "windows"]
//! DogTray v2：托盘启动器。
//! 只调用 Windows 自身 API。启动时读当前目录 start_config.ini，
//! 每个 [段] 的内容当作 cmd 批处理，无窗口启动；段与段之间隔 5 秒。
//! [main] 里 main= 指定的段是主程序，主程序完全退出后杀掉其它段再退出。
//! 退出时统一优雅关闭：先给有窗口的发 WM_CLOSE 等 5 秒，
//! 再给所有还没退出的程序发关机通知等 5 秒，最后才强行终止。
//! 无窗口程序也一同等，因为它可能依赖窗口程序。
//! 子程序树：直接 CreateProcess 启动 cmd.exe，cmd 是 DogTray 的子进程，
//! 批处理里再启动的程序是 cmd 的子进程。
//! 关键修法：只靠进程快照轮询会漏掉短命中间进程——
//! 比如 Chrome 启动时先起一个很快就退出的 launcher，真正的浏览器进程
//! 父进程指向那个已死的 launcher，快照轮询时 launcher 已不在列表里，
//! 浏览器就进不了跟踪集合，cmd 一退就被误判为整段已退出。
//! 所以每段都用一个 Windows 作业对象（Job Object）做内核级跟踪：
//! cmd 以 CREATE_SUSPENDED 起、还没跑任何命令前就先加入作业，
//! 之后它及所有后代（无论中间进程活多久）都会自动进作业，
//! 查作业进程列表就能拿到还活着的浏览器/渲染器等全部进程。
//! 快照轮询仍保留，兜底那些用 CREATE_BREAKAWAY_FROM_JOB 脱离作业的进程。
//!
//! v3：托盘左键打开主界面，主界面每个 [段] 一个按钮；点按钮打开该段的
//! 输出窗口（stdout + stderr 合并到同一管道）。每段输出在内存里只留
//! 最近 2MB，超了删前面、后面追加。输出窗口用 200ms 定时刷新：
//! 更新前先看滚动条是否到底，到底才跟随滚动，手动滚离则恢复原滚动位置。

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::fs::OpenOptions;
use std::io::Write;
use std::mem::{size_of, zeroed};
use std::path::PathBuf;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

type HWND = isize;
type HINSTANCE = isize;
type HICON = isize;
type HCURSOR = isize;
type HBRUSH = isize;
type HMENU = isize;
type HANDLE = isize;
type WPARAM = usize;
type LPARAM = isize;
type LRESULT = isize;
type UINT = u32;
type DWORD = u32;
type BOOL = i32;

const WM_DESTROY: UINT = 0x0002;
const WM_CLOSE: UINT = 0x0010;
const WM_QUERYENDSESSION: UINT = 0x0011;
const WM_ENDSESSION: UINT = 0x0016;
const WM_COMMAND: UINT = 0x0111;
const WM_TRAY: UINT = 0x8000 + 1;
const WM_APP_SHUTDOWN: UINT = 0x8000 + 2;
const WM_LBUTTONUP: UINT = 0x0202;
const WM_LBUTTONDBLCLK: UINT = 0x0203;
const WM_RBUTTONUP: UINT = 0x0205;
const WS_OVERLAPPEDWINDOW: DWORD = 0x00CF0000;
const CW_USEDEFAULT: i32 = i32::MIN;
const SW_HIDE: i32 = 0;
const SW_SHOW: i32 = 5;
const SW_RESTORE: i32 = 9;
const NIM_ADD: DWORD = 0;
const NIM_DELETE: DWORD = 2;
const NIF_MESSAGE: UINT = 0x00000001;
const NIF_ICON: UINT = 0x00000002;
const NIF_TIP: UINT = 0x00000004;
const MF_STRING: UINT = 0x00000000;
const TPM_RIGHTBUTTON: UINT = 0x00000002;
const TPM_RETURNCMD: UINT = 0x00000100;
const IDI_APPLICATION: usize = 32512;
const IDC_ARROW: usize = 32512;
const COLOR_WINDOW: isize = 5;
const ID_EXIT: u32 = 1001;
const CREATE_NO_WINDOW: DWORD = 0x08000000;
const CREATE_UNICODE_ENVIRONMENT: DWORD = 0x00000400;
const CREATE_SUSPENDED: DWORD = 0x00000004;
const JOB_OBJECT_BASIC_PROCESS_ID_LIST: i32 = 3;
const STARTF_USESHOWWINDOW: DWORD = 0x00000001;
const TH32CS_SNAPPROCESS: DWORD = 0x00000002;
const PROCESS_TERMINATE: DWORD = 0x00000001;
const PROCESS_QUERY_LIMITED_INFORMATION: DWORD = 0x00001000;
const MB_ICONERROR: UINT = 0x00000010;
const WS_CHILD: DWORD = 0x40000000;
const WS_VISIBLE: DWORD = 0x10000000;
const WS_VSCROLL: DWORD = 0x00200000;
const WS_HSCROLL: DWORD = 0x00100000;
const ES_MULTILINE: DWORD = 0x0004;
const ES_AUTOVSCROLL: DWORD = 0x0040;
const ES_AUTOHSCROLL: DWORD = 0x0080;
const ES_READONLY: DWORD = 0x0800;
const WM_SIZE: UINT = 0x0005;
const WM_TIMER: UINT = 0x0113;
const WM_VSCROLL: UINT = 0x0115;
const WM_SETFONT: UINT = 0x0030;
const EM_SETSEL: UINT = 0x00B1;
const EM_LINESCROLL: UINT = 0x00B6;
const EM_REPLACESEL: UINT = 0x00C2;
const EM_GETFIRSTVISIBLELINE: UINT = 0x00CE;
const SB_VERT: i32 = 1;
const SB_BOTTOM: WPARAM = 7;
const SIF_ALL: UINT = 0x17;
const STARTF_USESTDHANDLES: DWORD = 0x00000100;
const GENERIC_READ: DWORD = 0x80000000;
const FILE_SHARE_READ: DWORD = 0x00000001;
const FILE_SHARE_WRITE: DWORD = 0x00000002;
const OPEN_EXISTING: DWORD = 3;
const FILE_ATTRIBUTE_NORMAL: DWORD = 0x00000080;
const HANDLE_FLAG_INHERIT: DWORD = 0x00000001;
const SYSTEM_FIXED_FONT: i32 = 14;
const BUTTON_ID_BASE: u32 = 2000;
const OUTPUT_CACHE_LIMIT: usize = 2 * 1024 * 1024;

static SHARED: OnceLock<Arc<Shared>> = OnceLock::new();

#[repr(C)]
struct POINT {
    x: i32,
    y: i32,
}

#[repr(C)]
struct MSG {
    hwnd: HWND,
    message: UINT,
    w_param: WPARAM,
    l_param: LPARAM,
    time: DWORD,
    pt: POINT,
    l_private: DWORD,
}

#[repr(C)]
struct WNDCLASSEXW {
    cb_size: u32,
    style: u32,
    lpfn_wnd_proc: Option<unsafe extern "system" fn(HWND, UINT, WPARAM, LPARAM) -> LRESULT>,
    cb_cls_extra: i32,
    cb_wnd_extra: i32,
    h_instance: HINSTANCE,
    h_icon: HICON,
    h_cursor: HCURSOR,
    hbr_background: HBRUSH,
    lpsz_menu_name: *const u16,
    lpsz_class_name: *const u16,
    h_icon_sm: HICON,
}

#[repr(C)]
struct NOTIFYICONDATAW {
    cb_size: u32,
    h_wnd: HWND,
    u_id: UINT,
    u_flags: UINT,
    u_callback_message: UINT,
    h_icon: HICON,
    sz_tip: [u16; 128],
    dw_state: DWORD,
    dw_state_mask: DWORD,
    sz_info: [u16; 256],
    u_version: DWORD,
    sz_info_title: [u16; 64],
    dw_info_flags: DWORD,
    guid_item: [u8; 16],
    h_balloon_icon: HICON,
}

#[repr(C)]
struct SYSTEMTIME {
    w_year: u16,
    w_month: u16,
    w_day_of_week: u16,
    w_day: u16,
    w_hour: u16,
    w_minute: u16,
    w_second: u16,
    w_milliseconds: u16,
}

#[repr(C)]
struct STARTUPINFOW {
    cb: u32,
    lp_reserved: *mut u16,
    lp_desktop: *mut u16,
    lp_title: *mut u16,
    dw_x: u32,
    dw_y: u32,
    dw_x_size: u32,
    dw_y_size: u32,
    dw_x_count: u32,
    dw_y_count: u32,
    dw_fill_attribute: u32,
    dw_flags: u32,
    w_show_window: u16,
    cb_reserved2: u16,
    lp_reserved2: *mut u8,
    h_std_input: HANDLE,
    h_std_output: HANDLE,
    h_std_error: HANDLE,
}

#[repr(C)]
struct PROCESS_INFORMATION {
    h_process: HANDLE,
    h_thread: HANDLE,
    dw_process_id: u32,
    dw_thread_id: u32,
}

#[repr(C)]
struct PROCESSENTRY32W {
    dw_size: u32,
    cnt_usage: u32,
    th32_process_id: u32,
    th32_default_heap_id: usize,
    th32_module_id: u32,
    cnt_threads: u32,
    th32_parent_process_id: u32,
    pc_pri_class_base: i32,
    dw_flags: u32,
    sz_exe_file: [u16; 260],
}

#[repr(C)]
struct RECT {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

#[repr(C)]
struct SCROLLINFO {
    cb_size: u32,
    f_mask: u32,
    n_min: i32,
    n_max: i32,
    n_page: u32,
    n_pos: i32,
    n_track_pos: i32,
}

#[repr(C)]
struct SECURITY_ATTRIBUTES {
    n_length: u32,
    lp_security_descriptor: *mut c_void,
    b_inherit_handle: BOOL,
}

struct Section {
    name: String,
    root_pid: u32,
    root_handle: HANDLE,
    job_handle: HANDLE,
    pids: HashSet<u32>,
    // PID -> 创建时间（FILETIME 合成 u64，0 = 未能拿到，退回纯 PID 判断）。
    // PID 复用时创建时间一定不同，靠它区分“旧进程死了、新进程复用了 PID”。
    pid_creation: HashMap<u32, u64>,
    // PID -> 最后一次在快照里看到它还活着的 Unix 秒。
    // 用于判断“父进程死后才出现的子进程”：父进程最后一次活着之后
    // 超过宽限还新出现的、父进程 PID 相同的进程，不可能是原父进程的子进程。
    pid_last_seen_unix: HashMap<u32, u64>,
    // 自动锚点：本段根 cmd 的直接子进程的 exe 归属。
    // 非 Windows 目录的 exe 记目录（小写、带末尾 \），比如 ...\virtualbrowser\；
    // Windows 目录下的 exe 只记完整路径，不记目录，否则整个 System32 都会变成“自己人”。
    // 作业里只有路径落在锚点目录、或等于锚点完整路径的进程才算核心进程；
    // 其余（搜狗 SGTool、360 注入模块等外来进程）不阻塞退出、不被关闭/强杀。
    // 注意：自动锚点和手工锚点必须分开存。anchor_dirs.is_empty() 在三处
    // 表示“还没自动找到主程序目录”（核心兜底、孤儿锚点启用、路径兜底），
    // 如果手工锚点混进来，一配手工锚点这三处兜底就全部从一开始失效。
    anchor_dirs: HashSet<String>,
    anchor_files: HashSet<String>,
    // 手工锚点：[main] 或本段里 anchor_dirs= 配的目录，段创建时就存在。
    // 判断核心时同时查自动和手工两个集合，但“是否为空”的判断只看自动的
    // anchor_dirs，手工锚点只负责“多认几个自己人”，不改变自动机制行为。
    manual_anchor_dirs: HashSet<String>,
    // 已经记过锚点的直接子进程：PID -> 记锚点时的创建时间。
    // 同一进程实例每轮不再重复查 exe 路径；PID 被复用、创建时间变了时重新记。
    anchor_recorded: HashMap<u32, u64>,
    exited: bool,
    output_text: String,
    output_truncated: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FILETIME {
    dw_low_datetime: u32,
    dw_high_datetime: u32,
}

struct OutputWindowState {
    section_name: String,
    edit_hwnd: HWND,
    last_len: usize,
    last_truncated: u64,
}

struct Shared {
    sections: Mutex<Vec<Section>>,
    shutdown: AtomicBool,
    shutdown_done: AtomicBool,
    log_enabled: bool,
    log_path: PathBuf,
    main_name: String,
    hwnd: AtomicIsize,
    temp_bats: Mutex<Vec<PathBuf>>,
    exe_stem: String,
    tray_tip: String,
    button_sections: Vec<String>,
    // [main] 里配的全局手工锚点目录，作用于所有段，启动时就写进各段的 manual_anchor_dirs。
    global_anchor_dirs: Vec<String>,
    output_windows: Mutex<HashMap<HWND, OutputWindowState>>,
    // 主段根进程已退、后代还活着时，上次写诊断日志的时间；首次为 None，立即写一次。
    diag_last_main_blocker: Mutex<Option<Instant>>,
}

struct EnumCtx {
    tracked: *const HashSet<u32>,
    out: *mut Vec<HWND>,
}

#[link(name = "user32")]
extern "system" {
    fn GetModuleHandleW(lp_module_name: *const u16) -> HINSTANCE;
    fn LoadIconW(h_instance: HINSTANCE, lp_icon_name: *const u16) -> HICON;
    fn LoadCursorW(h_instance: HINSTANCE, lp_cursor_name: *const u16) -> HCURSOR;
    fn RegisterClassExW(lpwcx: *const WNDCLASSEXW) -> u16;
    fn CreateWindowExW(
        dw_ex_style: DWORD,
        lp_class_name: *const u16,
        lp_window_name: *const u16,
        dw_style: DWORD,
        x: i32,
        y: i32,
        n_width: i32,
        n_height: i32,
        h_wnd_parent: HWND,
        h_menu: HMENU,
        h_instance: HINSTANCE,
        lp_param: *const c_void,
    ) -> HWND;
    fn ShowWindow(h_wnd: HWND, n_cmd_show: i32) -> BOOL;
    fn SetForegroundWindow(h_wnd: HWND) -> BOOL;
    fn IsIconic(h_wnd: HWND) -> BOOL;
    fn GetMessageW(lp_msg: *mut MSG, h_wnd: HWND, w_msg_filter_min: UINT, w_msg_filter_max: UINT) -> i32;
    fn TranslateMessage(lp_msg: *const MSG) -> BOOL;
    fn DispatchMessageW(lp_msg: *const MSG) -> LRESULT;
    fn DefWindowProcW(h_wnd: HWND, msg: UINT, w_param: WPARAM, l_param: LPARAM) -> LRESULT;
    fn PostQuitMessage(n_exit_code: i32);
    fn GetCursorPos(lp_point: *mut POINT) -> BOOL;
    fn CreatePopupMenu() -> HMENU;
    fn AppendMenuW(h_menu: HMENU, u_flags: UINT, u_id_new_item: usize, lp_new_item: *const u16) -> BOOL;
    fn TrackPopupMenu(
        h_menu: HMENU,
        u_flags: UINT,
        x: i32,
        y: i32,
        n_reserved: i32,
        h_wnd: HWND,
        prc_rect: *const c_void,
    ) -> i32;
    fn PostMessageW(h_wnd: HWND, msg: UINT, w_param: WPARAM, l_param: LPARAM) -> BOOL;
    fn DestroyMenu(h_menu: HMENU) -> BOOL;
    fn MessageBoxW(h_wnd: HWND, lp_text: *const u16, lp_caption: *const u16, u_type: UINT) -> i32;
    fn EnumWindows(
        lp_enum_func: Option<unsafe extern "system" fn(HWND, LPARAM) -> BOOL>,
        l_param: LPARAM,
    ) -> BOOL;
    fn GetWindowThreadProcessId(h_wnd: HWND, lpdw_process_id: *mut DWORD) -> DWORD;
    fn SendMessageW(h_wnd: HWND, msg: UINT, w_param: WPARAM, l_param: LPARAM) -> LRESULT;
    fn SetWindowTextW(h_wnd: HWND, lp_string: *const u16) -> BOOL;
    fn GetClientRect(h_wnd: HWND, lp_rect: *mut RECT) -> BOOL;
    fn MoveWindow(h_wnd: HWND, x: i32, y: i32, n_width: i32, n_height: i32, b_repaint: BOOL) -> BOOL;
    fn SetTimer(h_wnd: HWND, n_id_event: usize, u_elapse: UINT, lp_timer_func: *const c_void) -> usize;
    fn KillTimer(h_wnd: HWND, u_id_event: usize) -> BOOL;
    fn GetScrollInfo(h_wnd: HWND, n_bar: i32, lp_scroll_info: *mut SCROLLINFO) -> BOOL;
    fn IsWindow(h_wnd: HWND) -> BOOL;
}

#[link(name = "gdi32")]
extern "system" {
    fn GetStockObject(fn_object: i32) -> isize;
}

#[link(name = "shell32")]
extern "system" {
    fn Shell_NotifyIconW(dw_message: DWORD, lp_data: *mut NOTIFYICONDATAW) -> BOOL;
}

#[link(name = "kernel32")]
extern "system" {
    fn GetModuleFileNameW(h_module: HINSTANCE, lp_filename: *mut u16, n_size: DWORD) -> DWORD;
    fn GetSystemDirectoryW(lp_buffer: *mut u16, u_size: UINT) -> UINT;
    fn GetLocalTime(lp_system_time: *mut SYSTEMTIME);
    fn CreateProcessW(
        lp_application_name: *const u16,
        lp_command_line: *mut u16,
        lp_process_attributes: *const c_void,
        lp_thread_attributes: *const c_void,
        b_inherit_handles: BOOL,
        dw_creation_flags: DWORD,
        lp_environment: *const c_void,
        lp_current_directory: *const u16,
        lp_startup_info: *const STARTUPINFOW,
        lp_process_information: *mut PROCESS_INFORMATION,
    ) -> BOOL;
    fn CloseHandle(h_object: HANDLE) -> BOOL;
    fn WaitForSingleObject(h_handle: HANDLE, dw_milliseconds: DWORD) -> DWORD;
    fn OpenProcess(dw_desired_access: DWORD, b_inherit_handle: BOOL, dw_process_id: DWORD) -> HANDLE;
    fn TerminateProcess(h_process: HANDLE, u_exit_code: UINT) -> BOOL;
    fn GetProcessTimes(
        h_process: HANDLE,
        lp_creation_time: *mut FILETIME,
        lp_exit_time: *mut FILETIME,
        lp_kernel_time: *mut FILETIME,
        lp_user_time: *mut FILETIME,
    ) -> BOOL;
    fn CreateToolhelp32Snapshot(dw_flags: DWORD, th32_process_id: DWORD) -> HANDLE;
    fn Process32FirstW(h_snapshot: HANDLE, lppe: *mut PROCESSENTRY32W) -> BOOL;
    fn Process32NextW(h_snapshot: HANDLE, lppe: *mut PROCESSENTRY32W) -> BOOL;
    fn CreateJobObjectW(lp_job_attributes: *const c_void, lp_name: *const u16) -> HANDLE;
    fn AssignProcessToJobObject(h_job: HANDLE, h_process: HANDLE) -> BOOL;
    fn GetWindowsDirectoryW(lp_buffer: *mut u16, u_size: UINT) -> UINT;
    fn GetEnvironmentVariableW(lp_name: *const u16, lp_buffer: *mut u16, n_size: DWORD) -> DWORD;
    fn ExpandEnvironmentStringsW(lp_src: *const u16, lp_dst: *mut u16, n_size: DWORD) -> DWORD;
    fn GetFullPathNameW(
        lp_file_name: *const u16,
        n_buffer_length: DWORD,
        lp_buffer: *mut u16,
        lp_file_part: *mut *mut u16,
    ) -> DWORD;
    fn QueryFullProcessImageNameW(
        h_process: HANDLE,
        dw_flags: DWORD,
        lp_exe_name: *mut u16,
        lpdw_size: *mut DWORD,
    ) -> BOOL;
    fn QueryInformationJobObject(
        h_job: HANDLE,
        job_object_information_class: i32,
        lp_job_object_information: *mut c_void,
        cb_job_object_information_length: DWORD,
        lp_return_length: *mut DWORD,
    ) -> BOOL;
    fn ResumeThread(h_thread: HANDLE) -> DWORD;
    fn CreatePipe(
        h_read_pipe: *mut HANDLE,
        h_write_pipe: *mut HANDLE,
        lp_pipe_attributes: *const SECURITY_ATTRIBUTES,
        n_size: DWORD,
    ) -> BOOL;
    fn SetHandleInformation(h_object: HANDLE, dw_mask: DWORD, dw_flags: DWORD) -> BOOL;
    fn ReadFile(
        h_file: HANDLE,
        lp_buffer: *mut c_void,
        n_number_of_bytes_to_read: DWORD,
        lp_number_of_bytes_read: *mut DWORD,
        lp_overlapped: *mut c_void,
    ) -> BOOL;
    fn CreateFileW(
        lp_file_name: *const u16,
        dw_desired_access: DWORD,
        dw_share_mode: DWORD,
        lp_security_attributes: *const SECURITY_ATTRIBUTES,
        dw_creation_disposition: DWORD,
        dw_flags_and_attributes: DWORD,
        h_template_file: HANDLE,
    ) -> HANDLE;
    fn MultiByteToWideChar(
        code_page: UINT,
        dw_flags: DWORD,
        lp_multi_byte_str: *const i8,
        cb_multi_byte: i32,
        lp_wide_char_str: *mut u16,
        cch_wide_char: i32,
    ) -> i32;
    fn WideCharToMultiByte(
        code_page: UINT,
        dw_flags: DWORD,
        lp_wide_char_str: *const u16,
        cch_wide_char: i32,
        lp_multi_byte_str: *mut u8,
        cb_multi_byte: i32,
        lp_default_char: *const u8,
        lp_used_default_char: *mut BOOL,
    ) -> i32;
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn decode_bytes(bytes: &[u8]) -> String {
    let mut b = bytes;
    if b.starts_with(&[0xEF, 0xBB, 0xBF]) {
        b = &b[3..];
    }
    if let Ok(s) = std::str::from_utf8(b) {
        return s.to_string();
    }
    // 不是合法 UTF-8 时按本机 ANSI（中文 Windows 通常是 GBK）解码。
    unsafe {
        let needed = MultiByteToWideChar(0, 0, b.as_ptr() as *const i8, b.len() as i32, null_mut(), 0);
        if needed <= 0 {
            return String::from_utf8_lossy(b).into_owned();
        }
        let mut buf = vec![0u16; needed as usize];
        let got = MultiByteToWideChar(0, 0, b.as_ptr() as *const i8, b.len() as i32, buf.as_mut_ptr(), needed);
        if got <= 0 {
            return String::from_utf8_lossy(b).into_owned();
        }
        String::from_utf16_lossy(&buf)
    }
}

fn decode_acp_bytes(bytes: &[u8]) -> String {
    unsafe {
        let needed = MultiByteToWideChar(0, 0, bytes.as_ptr() as *const i8, bytes.len() as i32, null_mut(), 0);
        if needed <= 0 {
            return String::from_utf8_lossy(bytes).into_owned();
        }
        let mut buf = vec![0u16; needed as usize];
        let got = MultiByteToWideChar(0, 0, bytes.as_ptr() as *const i8, bytes.len() as i32, buf.as_mut_ptr(), needed);
        if got <= 0 {
            return String::from_utf8_lossy(bytes).into_owned();
        }
        String::from_utf16_lossy(&buf)
    }
}

fn decode_line_bytes(bytes: &[u8]) -> String {
    let mut b = bytes;
    if b.starts_with(&[0xEF, 0xBB, 0xBF]) {
        b = &b[3..];
    }
    if let Ok(s) = std::str::from_utf8(b) {
        s.to_string()
    } else {
        decode_acp_bytes(b)
    }
}

/// 去掉 ANSI 颜色/控制序列，Edit 控件显示不了它们，留着只会变乱码。
fn strip_ansi(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\x1b' {
            if i + 1 >= chars.len() {
                break;
            }
            if chars[i + 1] == '[' {
                // CSI：一直吃到 0x40..=0x7E 的结束字节
                let mut j = i + 2;
                while j < chars.len() && !(('\x40'..='\x7e').contains(&chars[j])) {
                    j += 1;
                }
                i = if j < chars.len() { j + 1 } else { j };
            } else if chars[i + 1] == ']' {
                let mut j = i + 2;
                while j < chars.len() && chars[j] != '\x07' {
                    j += 1;
                }
                i = if j < chars.len() { j + 1 } else { j };
            } else {
                i += 2;
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn normalize_newlines(s: &str) -> String {
    s.replace("\r\n", "\n").replace('\r', "\n")
}

/// 每段输出只留最近 2MB：超了删前面整行，新信息追加到后面。
fn append_output(shared: &Shared, section_name: &str, decoded_line: &str) {
    if decoded_line.is_empty() {
        return;
    }
    let mut sections = match shared.sections.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let Some(sec) = sections.iter_mut().find(|s| s.name == section_name) else {
        return;
    };
    let new_len = decoded_line.len();
    if new_len >= OUTPUT_CACHE_LIMIT {
        // 单行本身就超 2MB：只留这行最后 2MB，且从行首/字符边界起。
        let start = decoded_line.len() - OUTPUT_CACHE_LIMIT;
        let mut s = start;
        while s < decoded_line.len() && !decoded_line.is_char_boundary(s) {
            s += 1;
        }
        sec.output_text = decoded_line[s..].to_string();
        sec.output_truncated += 1;
        return;
    }
    if sec.output_text.len() + new_len > OUTPUT_CACHE_LIMIT {
        let excess = sec.output_text.len() + new_len - OUTPUT_CACHE_LIMIT;
        // 先把 excess 对齐到下一个字符边界，再切片：直接按字节切片时，
        // excess 正好落在中文等多字节字符中间就会 panic（本工程 panic=abort，
        // 读线程一 panic 整个 DogTray 都会直接退出）。
        let mut aligned = excess;
        while aligned < sec.output_text.len() && !sec.output_text.is_char_boundary(aligned) {
            aligned += 1;
        }
        let drain_to = if let Some(pos) = sec.output_text[aligned..].find('\n') {
            aligned + pos + 1
        } else {
            aligned
        };
        if drain_to > 0 && drain_to <= sec.output_text.len() {
            sec.output_text.drain(..drain_to);
            sec.output_truncated += 1;
        }
    }
    sec.output_text.push_str(decoded_line);
}

/// 读一根管道：按行解码（每行单独试 UTF-8，不行按本机 ANSI/GBK），
/// 这样同一段里 bat 回显（GBK）和程序输出（UTF-8）混在一根管道里也不会互相弄乱。
fn output_reader_thread(shared: Arc<Shared>, section_name: String, read_handle: HANDLE) {
    let mut line_buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let mut read: DWORD = 0;
        let ok = unsafe {
            ReadFile(
                read_handle,
                chunk.as_mut_ptr() as *mut c_void,
                chunk.len() as DWORD,
                &mut read,
                null_mut(),
            )
        };
        if ok == 0 || read == 0 {
            break;
        }
        line_buf.extend_from_slice(&chunk[..read as usize]);
        // 一次性把已完整的行都吐出来；超长无换行行超过 64KB 也先吐，避免无限攒。
        loop {
            let newline_pos = line_buf.iter().position(|b| *b == b'\n');
            let take = match newline_pos {
                Some(p) => Some(p + 1),
                None if line_buf.len() > 65536 => Some(line_buf.len()),
                None => None,
            };
            let Some(n) = take else { break };
            let raw: Vec<u8> = line_buf.drain(..n).collect();
            let text = normalize_newlines(&strip_ansi(&decode_line_bytes(&raw)));
            let mut line = text;
            if !line.ends_with('\n') {
                line.push('\n');
            }
            append_output(&shared, &section_name, &line);
        }
    }
    if !line_buf.is_empty() {
        let text = normalize_newlines(&strip_ansi(&decode_line_bytes(&line_buf)));
        let mut line = text;
        if !line.ends_with('\n') {
            line.push('\n');
        }
        append_output(&shared, &section_name, &line);
    }
    unsafe {
        CloseHandle(read_handle);
    }
}

fn wide_to_acp(s: &str) -> Vec<u8> {
    let w = wide(s);
    let len_no_null = if w.is_empty() { 0 } else { (w.len() - 1) as i32 };
    unsafe {
        let needed = WideCharToMultiByte(0, 0, w.as_ptr(), len_no_null, null_mut(), 0, null(), null_mut());
        if needed <= 0 {
            return s.as_bytes().to_vec();
        }
        let mut buf = vec![0u8; needed as usize];
        let got = WideCharToMultiByte(0, 0, w.as_ptr(), len_no_null, buf.as_mut_ptr(), needed, null(), null_mut());
        if got <= 0 {
            return s.as_bytes().to_vec();
        }
        buf
    }
}

fn strip_quotes(v: &str) -> String {
    let t = v.trim();
    if t.len() >= 2 {
        let first = t.chars().next().unwrap();
        let last = t.chars().last().unwrap();
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return t[1..t.len() - 1].to_string();
        }
    }
    t.to_string()
}

struct Config {
    // (段名, 脚本, 该段手工锚点目录)
    sections: Vec<(String, String, Vec<String>)>,
    main_name: String,
    output_log: bool,
    title: Option<String>,
    // [main] 里配的全局手工锚点目录，作用于所有段。
    global_anchor_dirs: Vec<String>,
}

/// 解析 anchor_dirs= 的值：分号分隔，多个值、多个行都累积。
/// 空值、空项直接跳过，不会记成空锚点。
fn parse_anchor_dirs_value(val: &str, out: &mut Vec<String>) {
    for part in val.split(';') {
        if let Some(dir) = normalize_anchor_dir(part) {
            if !out.contains(&dir) {
                out.push(dir);
            }
        }
    }
}

fn parse_config(text: &str) -> Config {
    let mut ordered: Vec<(String, Vec<String>)> = Vec::new();
    let mut cur_name: Option<String> = None;
    let mut cur_lines: Vec<String> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') && trimmed.len() >= 2 {
            if let Some(n) = cur_name.take() {
                ordered.push((n, std::mem::take(&mut cur_lines)));
            }
            cur_name = Some(trimmed[1..trimmed.len() - 1].trim().to_string());
        } else if cur_name.is_some() {
            cur_lines.push(line.to_string());
        }
    }
    if let Some(n) = cur_name.take() {
        ordered.push((n, cur_lines));
    }

    let mut main_name = String::new();
    let mut output_log = false;
    let mut title: Option<String> = None;
    let mut global_anchor_dirs: Vec<String> = Vec::new();
    let mut sections: Vec<(String, String, Vec<String>)> = Vec::new();
    for (name, lines) in ordered {
        if name.eq_ignore_ascii_case("main") {
            for line in lines {
                let t = line.trim();
                if t.is_empty() {
                    continue;
                }
                if let Some(eq) = t.find('=') {
                    let key = t[..eq].trim().to_ascii_lowercase();
                    let val = strip_quotes(t[eq + 1..].trim());
                    match key.as_str() {
                        "main" => main_name = val.to_ascii_lowercase(),
                        "output_log" => {
                            output_log = matches!(val.to_ascii_lowercase().as_str(), "true" | "1" | "yes")
                        }
                        "title" => {
                            if !val.is_empty() {
                                title = Some(val);
                            }
                        }
                        "anchor_dirs" => parse_anchor_dirs_value(&val, &mut global_anchor_dirs),
                        _ => {}
                    }
                }
            }
        } else {
            // 各段里 anchor_dirs= 是配置行，不是批处理命令：
            // 提取出来当手工锚点，不放进脚本，否则 cmd 会把它当命令执行。
            // 只有行首就是 anchor_dirs= 才提取，set anchor_dirs= 这种
            // 批处理里的 set 命令不受影响。
            let mut sec_anchor_dirs: Vec<String> = Vec::new();
            let mut script_lines: Vec<String> = Vec::new();
            for line in lines {
                let t = line.trim();
                if let Some(eq) = t.find('=') {
                    let key = t[..eq].trim().to_ascii_lowercase();
                    if key == "anchor_dirs" {
                        let val = strip_quotes(t[eq + 1..].trim());
                        parse_anchor_dirs_value(&val, &mut sec_anchor_dirs);
                        continue;
                    }
                }
                script_lines.push(line);
            }
            let script = script_lines.join("\r\n");
            sections.push((name, script, sec_anchor_dirs));
        }
    }
    Config {
        sections,
        main_name,
        output_log,
        title,
        global_anchor_dirs,
    }
}

fn timestamp() -> String {
    unsafe {
        let mut st: SYSTEMTIME = zeroed();
        GetLocalTime(&mut st);
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            st.w_year, st.w_month, st.w_day, st.w_hour, st.w_minute, st.w_second
        )
    }
}

fn log_msg(shared: &Shared, msg: &str) {
    if !shared.log_enabled {
        return;
    }
    let line = format!("[{}] {}\r\n", timestamp(), msg);
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&shared.log_path) {
        let _ = f.write_all(line.as_bytes());
        let _ = f.flush();
    }
}

/// 诊断日志：受 output_log 控制，和普通日志一样。
/// output_log=false 时不写，output_log=true 时才写。
/// 触发条件本身不变：只在主段根进程已退、后代还活着这种反常情况下才写，
/// 正常使用（浏览器开着）不会写，避免把日志打爆。
fn log_diag(shared: &Shared, msg: &str) {
    if !shared.log_enabled {
        return;
    }
    let line = format!("[{}] 诊断: {}\r\n", timestamp(), msg);
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&shared.log_path) {
        let _ = f.write_all(line.as_bytes());
        let _ = f.flush();
    }
}

fn creation_time_from_handle(handle: HANDLE) -> Option<u64> {
    if handle == 0 || handle == -1 {
        return None;
    }
    unsafe {
        let mut creation: FILETIME = FILETIME { dw_low_datetime: 0, dw_high_datetime: 0 };
        let mut exit: FILETIME = FILETIME { dw_low_datetime: 0, dw_high_datetime: 0 };
        let mut kernel: FILETIME = FILETIME { dw_low_datetime: 0, dw_high_datetime: 0 };
        let mut user: FILETIME = FILETIME { dw_low_datetime: 0, dw_high_datetime: 0 };
        if GetProcessTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user) == 0 {
            return None;
        }
        Some(((creation.dw_high_datetime as u64) << 32) | (creation.dw_low_datetime as u64))
    }
}

fn exit_time_from_handle(handle: HANDLE) -> Option<u64> {
    if handle == 0 || handle == -1 {
        return None;
    }
    unsafe {
        let mut creation: FILETIME = FILETIME { dw_low_datetime: 0, dw_high_datetime: 0 };
        let mut exit: FILETIME = FILETIME { dw_low_datetime: 0, dw_high_datetime: 0 };
        let mut kernel: FILETIME = FILETIME { dw_low_datetime: 0, dw_high_datetime: 0 };
        let mut user: FILETIME = FILETIME { dw_low_datetime: 0, dw_high_datetime: 0 };
        if GetProcessTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user) == 0 {
            return None;
        }
        let t = ((exit.dw_high_datetime as u64) << 32) | (exit.dw_low_datetime as u64);
        if t == 0 {
            None
        } else {
            Some(t)
        }
    }
}

fn process_creation_time(pid: u32) -> Option<u64> {
    // 每次现查，不缓存：只按 PID 做键的缓存会在 PID 1 秒内被复用时
    // 返回旧进程的创建时间，is_same_process 就会把新进程误认为旧进程。
    // 虽然强杀前可以绕过缓存现查，但 remaining_pids 作业无效回退分支
    // 的结果决定给谁发 WM_CLOSE 和关机通知，同样可能误发给复用 PID 的
    // 无关程序窗口。省下的只是每轮几十次 OpenProcess，不值得冒这个险。
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h == 0 || h == -1 {
            return None;
        }
        let t = creation_time_from_handle(h);
        CloseHandle(h);
        t
    }
}

/// 同一个 PID 现在是不是当初记录的那个进程。
/// recorded==0 表示当初没拿到创建时间，退回纯 PID 判断；
/// 当前打不开（已在快照和打开之间退了）按已死算；
/// 创建时间对不上就是 PID 被复用了，按已死算，绝不操作它。
fn is_same_process(pid: u32, recorded: u64) -> bool {
    if recorded == 0 {
        return true;
    }
    match process_creation_time(pid) {
        Some(now) => now == recorded,
        None => false,
    }
}

/// 第一次发现 PID 时记下创建时间；已记录且创建时间变了，
/// 说明是父进程/作业新起的、复用了旧 PID 的子进程，更新成新的。
/// 注意：这个“更新”只对作业列表里的 PID 安全（作业是内核成员关系），
/// 快照父子链里已在集合里的 PID 绝不能靠它来“合法化”，
/// 否则误加的进程会在这里把创建时间改成它自己的，之后就再也查不出来了。
/// 所以下面快照加入时另有独立的创建时间顺序检查，不走这条更新路径。
fn record_pid_creation(sec: &mut Section, pid: u32) {
    if let Some(&recorded) = sec.pid_creation.get(&pid) {
        if recorded != 0 {
            if let Some(now) = process_creation_time(pid) {
                if now != recorded {
                    sec.pid_creation.insert(pid, now);
                }
            }
        }
        return;
    }
    sec.pid_creation.insert(pid, process_creation_time(pid).unwrap_or(0));
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn filetime_to_unix(ft: u64) -> u64 {
    // FILETIME 是 100 纳秒，自 1601 年起；Unix 纪元差 11644473600 秒。
    const EPOCH_DIFF_100NS: u64 = 116444736000000000;
    if ft < EPOCH_DIFF_100NS {
        0
    } else {
        (ft - EPOCH_DIFF_100NS) / 10_000_000
    }
}

/// 快照父子链加入检查：子进程能不能是父进程的子进程。
/// 必须同时满足：
/// 1) 子进程创建时间晚于父进程创建时间（子进程比父进程早出生就不可能是它的子进程，
///    典型就是 v2rayN 那种：无关进程创建时间早于后来复用其父进程 PID 的浏览器子进程）；
/// 2) 父进程还活着时自动满足；父进程已死时，子进程创建时间
///    不能晚于父进程最后一次被看到还活着的时间 + 5 秒宽限
///    （宽限是轮询间隔和秒级截断的误差，父进程死后几分钟、几小时才出现的新进程一定不满足）。
/// 任一条件不满足就不加入，宁可漏掉一个极罕见的脱离作业孤儿，
/// 也不把一整棵无关进程树吸进来、让主段永远退不出去、甚至强杀错进程。
/// 作业列表加入不走这个检查，因为作业是内核成员关系，不是靠父进程 PID 猜的。
fn can_add_via_parent(
    parent_creation: u64,
    parent_alive: bool,
    parent_last_seen_unix: Option<u64>,
    child_creation: u64,
) -> bool {
    if parent_creation == 0 || child_creation == 0 {
        return false;
    }
    // 同样用原始 FILETIME 比较，不转秒：同一秒内创建的父子进程很常见，
    // 转秒会把它们误判为复用而漏加。
    if child_creation <= parent_creation {
        return false;
    }
    if parent_alive {
        return true;
    }
    match parent_last_seen_unix {
        Some(last) => filetime_to_unix(child_creation) <= last.saturating_add(5),
        None => false,
    }
}

fn windows_dir_lower() -> String {
    static WIN_DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    WIN_DIR
        .get_or_init(|| unsafe {
            let mut buf = vec![0u16; 260];
            let n = GetWindowsDirectoryW(buf.as_mut_ptr(), buf.len() as UINT);
            if n == 0 || n as usize >= buf.len() {
                return "c:\\windows\\".to_string();
            }
            let mut s = String::from_utf16_lossy(&buf[..n as usize]).to_lowercase();
            if !s.ends_with('\\') {
                s.push('\\');
            }
            s
        })
        .clone()
}

fn user_profile_dir_lower() -> Option<String> {
    static PROFILE_DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PROFILE_DIR
        .get_or_init(|| unsafe {
            let name = wide("USERPROFILE");
            let mut buf = vec![0u16; 1024];
            let n = GetEnvironmentVariableW(name.as_ptr(), buf.as_mut_ptr(), buf.len() as DWORD);
            if n == 0 || n as usize >= buf.len() {
                return None;
            }
            let mut s = String::from_utf16_lossy(&buf[..n as usize])
                .replace('/', "\\")
                .to_lowercase();
            if !s.ends_with('\\') {
                s.push('\\');
            }
            Some(s)
        })
        .clone()
}

/// 锚点目录是不是过宽的根目录：盘符根（d:\）、用户目录根（%USERPROFILE%\）、
/// UNC 共享根（\\server\share\）。这种目录一旦记成锚点，整个盘、整个用户
/// 目录、整个共享里的程序都会变成核心，包括装在那里的输入法。
fn is_broad_root_dir(dir: &str) -> bool {
    let bytes = dir.as_bytes();
    if dir.len() == 3 && bytes[1] == b':' && bytes[2] == b'\\' && bytes[0].is_ascii_alphabetic() {
        return true;
    }
    if let Some(profile) = user_profile_dir_lower() {
        if dir == profile {
            return true;
        }
    }
    if dir.starts_with("\\\\") {
        // UNC 根：\\server\share\，除开头的 \\ 外只有 server、share 两段。
        let rest = &dir[2..];
        let parts: Vec<&str> = rest.split('\\').filter(|p| !p.is_empty()).collect();
        if parts.len() == 2 && dir.ends_with('\\') {
            return true;
        }
    }
    false
}

/// 手工锚点目录规范化：去引号、/ 转 \、小写、补末尾 \、相对路径按当前目录展开。
/// 返回 None 表示这行写了空值，调用方直接跳过。
fn normalize_anchor_dir(raw: &str) -> Option<String> {
    let s = strip_quotes(raw.trim()).trim().to_string();
    if s.is_empty() {
        return None;
    }
    unsafe {
        // 1) 展开环境变量：%LOCALAPPDATA%\VirtualBrowser 这种批处理里
        //    常见的写法，不展开就永远匹配不上规范化后的 exe 路径。
        let expanded = {
            let src = wide(&s);
            let mut buf = vec![0u16; 4096];
            let n = ExpandEnvironmentStringsW(src.as_ptr(), buf.as_mut_ptr(), buf.len() as DWORD);
            if n == 0 || n as usize >= buf.len() {
                s.clone()
            } else {
                String::from_utf16_lossy(&buf[..n as usize - 1])
            }
        };
        // 2) GetFullPathNameW 规范化：处理相对路径、. 和 ..、
        //    以及 D:foo 这种不带反斜杠的盘符相对写法。
        //    它按当前目录解析相对路径，不要求路径真实存在。
        //    注意：..\tools 手工拼当前目录会得到 d:\dog\..\tools\，
        //    而 QueryFullProcessImageNameW 返回的是 d:\tools\...，
        //    前缀匹配永远失败，必须在这里规范化掉 ..。
        let full = {
            let src = wide(&expanded);
            let mut buf = vec![0u16; 4096];
            let mut part: *mut u16 = std::ptr::null_mut();
            let n = GetFullPathNameW(
                src.as_ptr(),
                buf.len() as DWORD,
                buf.as_mut_ptr(),
                &mut part,
            );
            if n == 0 || n as usize >= buf.len() {
                expanded.clone()
            } else {
                String::from_utf16_lossy(&buf[..n as usize])
            }
        };
        let mut lower = full.replace('/', "\\").to_lowercase();
        if !lower.ends_with('\\') {
            lower.push('\\');
        }
        Some(lower)
    }
}

fn process_exe_path(pid: u32) -> Option<String> {
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h == 0 || h == -1 {
            return None;
        }
        let mut buf = vec![0u16; 1024];
        let mut size: DWORD = buf.len() as DWORD;
        let ok = QueryFullProcessImageNameW(h, 0, buf.as_mut_ptr(), &mut size);
        CloseHandle(h);
        if ok == 0 {
            return None;
        }
        Some(String::from_utf16_lossy(&buf[..size as usize]))
    }
}

/// exe 路径缓存：键是 PID + 创建时间，进程实例不变就不再 OpenProcess。
/// 创建时间从 sec.pid_creation 里取（跟踪时已经拿到过），查缓存本身不需要
/// 再打开进程；创建时间未知（0）时不缓存，每次现查。
/// 缓存是全局的，因为 PID + 创建时间在系统范围内唯一标识一个进程实例，
/// 不同段查到同一进程时路径一定相同。
fn cached_exe_path(sec: &Section, pid: u32) -> Option<String> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<(u32, u64), Option<String>>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let creation = sec.pid_creation.get(&pid).copied().unwrap_or(0);
    if creation != 0 {
        if let Ok(map) = cache.lock() {
            if let Some(cached) = map.get(&(pid, creation)) {
                return cached.clone();
            }
        }
    }
    let path = process_exe_path(pid);
    if creation != 0 {
        if let Ok(mut map) = cache.lock() {
            // 缓存条数只跟本轮遇到过的进程实例数相当，不会无限增长到失控，
            // DogTray 一轮通常只有几十个进程；超过 10000 条时清空重建，
            // 避免长期运行后旧 PID 复用条目越积越多。
            if map.len() > 10000 {
                map.clear();
            }
            map.insert((pid, creation), path.clone());
        }
    }
    path
}

fn record_anchor(sec: &mut Section, exe_path: &str) {
    let lower = exe_path.to_lowercase();
    let win = windows_dir_lower();
    if lower.starts_with(&win) {
        // Windows 目录下的 exe 只记完整路径，不记目录。
        sec.anchor_files.insert(lower);
    } else if let Some(pos) = lower.rfind('\\') {
        let dir = lower[..=pos].to_string();
        if is_broad_root_dir(&dir) {
            // 盘符根、用户目录根、UNC 共享根太宽：记目录会把整个盘 /
            // 整个用户目录里的输入法都算成核心，降级为只记完整路径。
            sec.anchor_files.insert(lower);
        } else {
            sec.anchor_dirs.insert(dir);
        }
    }
}

/// 作业/快照成员是不是本段的核心进程。
/// - 父进程是根 cmd 的直接子进程，一定是核心，但必须同时满足：
///   1) 创建时间用原始 FILETIME（100 纳秒精度）比较，子进程晚于根 cmd；
///      转成秒再比较会把同一秒内创建的 cmd 和子进程误判为复用，锚点记不上；
///   2) 根 cmd 已退出时，子进程创建时间必须早于根 cmd 退出时间；
///      否则根 cmd 退出后 PID 被复用、新进程拉起的 SGTool 也会被误判为核心。
///      root_exit_time == 0 表示根 cmd 还没退、或拿不到退出时间，此时不设上限。
/// - exe 路径落在锚点目录、或等于锚点完整路径的，是核心；
/// - exe 路径拿不到的，按核心算（保守：宁可多等一会，也不提前判定退出）；
/// - 其余的都是外来进程：搜狗 SGTool、360 注入模块等，它们不在启动程序的目录里。
/// parent_ppid 为 None 时表示快照里没找到该进程的父进程信息，按路径判断。
/// 兜底：只有自动的 anchor_dirs 为空时才全部按核心算。Windows 目录下的完整路径
/// 锚点（xcopy、timeout 之类）不能代表主程序，不能让兜底失效，
/// 否则启动器早死、主程序目录还没记下来时，真正的浏览器会被判成外来。
/// 手工锚点不参与这个判断：即使配了手工锚点，自动锚点还没记上时兜底仍生效，
/// 否则 [main] 里为某个 python 程序配了手工锚点，chrome 段遇到启动器早死
/// 时孤儿规则和兜底都会失效，真正的浏览器会被判成外来而提前退出。
fn is_core_process(
    sec: &Section,
    pid: u32,
    parent_ppid: Option<u32>,
    exe_path: Option<&str>,
    root_exit_time: u64,
) -> bool {
    if let Some(ppid) = parent_ppid {
        if ppid == sec.root_pid {
            let root_creation = sec.pid_creation.get(&sec.root_pid).copied().unwrap_or(0);
            let child_creation = sec.pid_creation.get(&pid).copied().unwrap_or(0);
            if root_creation != 0 && child_creation != 0 {
                if child_creation > root_creation
                    && (root_exit_time == 0 || child_creation < root_exit_time)
                {
                    return true;
                }
                // 创建时间对不上（PID 复用、直接子进程是假的，或子进程
                // 在根 cmd 退出后才出现），不按直接子进程算，继续按路径判断。
            } else if root_creation == 0 {
                // 根 cmd 创建时间没拿到，无法做创建时间校验，仍按直接子进程算核心，
                // 因为父进程 PID 本身就是最直接的证据，且 PID 复用概率极低。
                return true;
            }
        }
    }
    let Some(path) = exe_path else {
        return true;
    };
    let lower = path.to_lowercase();
    if sec.anchor_files.contains(&lower) {
        return true;
    }
    if sec.anchor_dirs.iter().any(|d| lower.starts_with(d)) {
        return true;
    }
    if sec.manual_anchor_dirs.iter().any(|d| lower.starts_with(d)) {
        return true;
    }
    if sec.anchor_dirs.is_empty() {
        return true;
    }
    false
}

fn root_exit_time_of(sec: &Section) -> u64 {
    if sec.root_handle == 0 || sec.root_handle == -1 {
        return 0;
    }
    // GetProcessTimes 在进程还活着时返回的退出时间是“未定义”的，
    // 文档不保证是 0，所以必须先确认根进程已经退出再取，没退出就返回 0。
    let exited = unsafe { WaitForSingleObject(sec.root_handle, 0) == 0 };
    if !exited {
        return 0;
    }
    exit_time_from_handle(sec.root_handle).unwrap_or(0)
}

fn snapshot_pid_names() -> HashMap<u32, String> {
    let mut map = HashMap::new();
    unsafe {
        let h = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if h == -1 || h == 0 {
            return map;
        }
        let mut entry: PROCESSENTRY32W = zeroed();
        entry.dw_size = size_of::<PROCESSENTRY32W>() as u32;
        if Process32FirstW(h, &mut entry) != 0 {
            loop {
                let end = entry.sz_exe_file.iter().position(|c| *c == 0).unwrap_or(entry.sz_exe_file.len());
                let name = String::from_utf16_lossy(&entry.sz_exe_file[..end]);
                map.insert(entry.th32_process_id, name);
                if Process32NextW(h, &mut entry) == 0 {
                    break;
                }
            }
        }
        CloseHandle(h);
    }
    map
}

fn fmt_pid_list(pids: &[u32], names: &HashMap<u32, String>) -> String {
    let mut parts: Vec<String> = Vec::new();
    for pid in pids {
        if let Some(name) = names.get(pid) {
            parts.push(format!("{}:{}", pid, name));
        } else {
            parts.push(format!("{}", pid));
        }
    }
    parts.join(", ")
}

fn exe_stem() -> String {
    unsafe {
        let mut buf = vec![0u16; 1024];
        let len = GetModuleFileNameW(0, buf.as_mut_ptr(), 1024);
        if len == 0 {
            return "DogTray".to_string();
        }
        let path = String::from_utf16_lossy(&buf[..len as usize]);
        let file = path.rsplit('\\').next().unwrap_or(&path);
        if let Some(pos) = file.rfind('.') {
            file[..pos].to_string()
        } else {
            file.to_string()
        }
    }
}

fn show_missing_config(stem: &str) {
    let text = wide("缺少配置文件 start_config.ini");
    let title = wide(stem);
    unsafe {
        MessageBoxW(0, text.as_ptr(), title.as_ptr(), MB_ICONERROR);
    }
}

/// 查一个作业对象里当前还活着的进程 PID 列表。
/// 结构：u32 已分配数，u32 列表中数，之后是 usize（x64 为 u64）PID 数组。
fn query_job_pids(job_handle: HANDLE) -> Vec<u32> {
    if job_handle == 0 || job_handle == -1 {
        return Vec::new();
    }
    unsafe {
        // 先按 512 个进程的容量查，不够再按返回的已分配数扩容重查。
        let mut capacity: usize = 512;
        for _ in 0..2 {
            let buf_size = 8 + capacity * size_of::<usize>();
            let mut buf = vec![0u8; buf_size];
            let mut ret_len: DWORD = 0;
            let ok = QueryInformationJobObject(
                job_handle,
                JOB_OBJECT_BASIC_PROCESS_ID_LIST,
                buf.as_mut_ptr() as *mut c_void,
                buf_size as DWORD,
                &mut ret_len,
            );
            if ok == 0 {
                return Vec::new();
            }
            let assigned = u32::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
            let in_list = u32::from_ne_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
            if in_list < assigned && capacity < assigned {
                capacity = assigned;
                continue;
            }
            let mut out = Vec::new();
            for i in 0..in_list {
                let off = 8 + i * size_of::<usize>();
                if off + size_of::<usize>() > buf.len() {
                    break;
                }
                // x64 下 PID 数组元素是 u64，直接按指针读。
                let pid_usize = *(buf.as_ptr().add(off) as *const usize);
                out.push(pid_usize as u32);
            }
            return out;
        }
        Vec::new()
    }
}

fn snapshot_processes() -> (HashSet<u32>, Vec<(u32, u32)>) {
    let mut set = HashSet::new();
    let mut list = Vec::new();
    unsafe {
        let h = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if h == -1 || h == 0 {
            return (set, list);
        }
        let mut entry: PROCESSENTRY32W = zeroed();
        entry.dw_size = size_of::<PROCESSENTRY32W>() as u32;
        if Process32FirstW(h, &mut entry) != 0 {
            loop {
                set.insert(entry.th32_process_id);
                list.push((entry.th32_process_id, entry.th32_parent_process_id));
                if Process32NextW(h, &mut entry) == 0 {
                    break;
                }
            }
        }
        CloseHandle(h);
    }
    (set, list)
}

fn update_descendants(shared: &Shared) {
    let (alive_set, list) = snapshot_processes();
    // 先把每段作业对象的进程列表取出来，避免在锁里调内核查询时
    // 借用问题：job 句柄是整数，复制出来即可。
    let job_handles: Vec<HANDLE> = match shared.sections.lock() {
        Ok(sections) => sections.iter().map(|s| s.job_handle).collect(),
        Err(_) => return,
    };
    let job_lists: Vec<Vec<u32>> = job_handles.iter().map(|h| query_job_pids(*h)).collect();
    let mut sections = match shared.sections.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let now_unix = unix_now();
    for (idx, sec) in sections.iter_mut().enumerate() {
        // 本次快照里还活着的已跟踪进程，更新最后一次活着的时间。
        // 死进程不更新，last_seen 就停在它最后一次活着的那次轮询。
        for pid in sec.pids.clone().iter() {
            if alive_set.contains(pid) {
                sec.pid_last_seen_unix.insert(*pid, now_unix);
            }
        }
        // 作业对象是内核级跟踪：中间进程活多久都不影响，
        // 它的子进程一创建就自动进作业，查作业列表就能直接拿到。
        // 作业成员关系是内核保证的，不用、也不走快照父子链的创建时间检查。
        if idx < job_lists.len() {
            for pid in &job_lists[idx] {
                if sec.pids.insert(*pid) {
                    record_pid_creation(sec, *pid);
                } else {
                    // 已在集合里：若创建时间变了，说明是新起的、复用旧 PID 的子进程。
                    // 只有作业列表能这样更新，因为作业成员关系本身就是证据。
                    record_pid_creation(sec, *pid);
                }
                sec.pid_last_seen_unix.insert(*pid, now_unix);
            }
        }
        // 锚点记录：轮询作业成员时记两种锚点，分两阶段，先直接子进程、后孤儿。
        // 1) 父进程是本段根 cmd 的直接子进程，就记下它的 exe 归属。
        //    判断时用原始 FILETIME 比较，子进程必须晚于根 cmd，
        //    且根 cmd 已退出时必须早于根 cmd 退出时间，否则是 PID 复用
        //    造成的假直接子进程，不记。同一秒内创建很常见，转秒比较会
        //    把 xray、VirtualBrowser 这种几乎和 cmd 同时启动的程序漏掉。
        //    根 cmd 死了以后，快照里孤儿的父进程 PID 仍然是根 cmd，所以即使
        //    根 cmd 已退，主程序还活着时锚点仍能记下来。
        // 2) 孤儿锚点：父进程已经不在了、而且从未出现在 sec.pids 里
        //    （说明父进程是个没来得及观察到的短命中间进程，比如 Chrome launcher），
        //    就把这个作业成员本身也当作锚点，记下它的目录。
        //    SGTool 不会命中这条，因为它的父进程是浏览器，浏览器早已被观察到、
        //    一定在 sec.pids 里。
        //    孤儿规则只在 anchor_dirs 还为空时生效，且必须在直接子进程
        //    锚点记完之后再判：这条本来就是为了“启动器早死、主程序目录还没记上”
        //    兜底的，一旦记到了任何一个非 Windows 目录锚点，就说明主程序
        //    已经找到了，之后不再接受孤儿锚点。否则外来软件先起一个短命
        //    中转进程、再拉起常驻进程（输入法更新器、安全软件中转），
        //    常驻进程的目录也会被记成锚点，SGTool 又会变回核心。
        // 已经记过锚点的直接子进程（同一 PID + 创建时间）本轮跳过，不再查路径。
        {
            let root_creation = sec.pid_creation.get(&sec.root_pid).copied().unwrap_or(0);
            let root_exit = root_exit_time_of(sec);
            let record_one = |sec: &mut Section, pid: u32| {
                let child_creation = sec.pid_creation.get(&pid).copied().unwrap_or(0);
                if sec.anchor_recorded.get(&pid).copied().unwrap_or(0) == child_creation
                    && child_creation != 0
                {
                    return;
                }
                if let Some(path) = cached_exe_path(sec, pid) {
                    record_anchor(sec, &path);
                    sec.anchor_recorded.insert(pid, child_creation);
                }
            };
            // 阶段一：直接子进程。
            for (pid, ppid) in &list {
                if *pid == sec.root_pid || *ppid != sec.root_pid {
                    continue;
                }
                let in_job = idx < job_lists.len() && job_lists[idx].contains(pid);
                if !in_job && !sec.pids.contains(pid) {
                    continue;
                }
                let child_creation = sec.pid_creation.get(pid).copied().unwrap_or(0);
                if root_creation != 0 && child_creation != 0 {
                    if child_creation <= root_creation {
                        continue;
                    }
                    if root_exit != 0 && child_creation >= root_exit {
                        continue;
                    }
                }
                record_one(sec, *pid);
            }
            // 阶段二：孤儿锚点，只在直接子进程记完后 anchor_dirs 仍为空时。
            // 每记一个就重新判断：孤儿本来就是“主程序目录还没记上”才用的
            // 兜底，一轮里只需要一个，记上一个目录锚点就立即停，
            // 避免同一轮多个孤儿把多个外来目录都记进来。
            // 注意：Windows 目录孤儿只记完整路径锚点，不记目录，
            // anchor_dirs 仍为空时继续循环是符合这条判断的。
            if sec.anchor_dirs.is_empty() {
                for (pid, ppid) in &list {
                    if *pid == sec.root_pid || *ppid == 0 {
                        continue;
                    }
                    let in_job = idx < job_lists.len() && job_lists[idx].contains(pid);
                    if !in_job && !sec.pids.contains(pid) {
                        continue;
                    }
                    if alive_set.contains(ppid) || sec.pids.contains(ppid) {
                        continue;
                    }
                    record_one(sec, *pid);
                    if !sec.anchor_dirs.is_empty() {
                        break;
                    }
                }
            }
        }
        // 快照父子链：只作作业的兜底（脱离作业的进程），且加入时必须通过
        // 创建时间顺序检查。以前这里只看“父进程 PID 在集合里”，
        // 死父进程 PID 一直留着当标记，任何父进程 PID 等于它的无关进程
        // （比如 v2rayN，它的父进程 PID 后来被浏览器子进程复用了）
        // 都会被吸进来，再带进它整棵子树，导致主段永远退不出去。
        let mut changed = true;
        while changed {
            changed = false;
            for (pid, ppid) in &list {
                if !sec.pids.contains(ppid) || sec.pids.contains(pid) {
                    continue;
                }
                // 父进程创建时间：已记录的直接用；没记录、但父进程现在还活着的，
                // 现在打开一次拿到（父进程活着，现在拿到的就是它自己的时间）；
                // 没记录、父进程现在也死了的，无法验证，直接不加。
                let parent_creation = match sec.pid_creation.get(ppid).copied() {
                    Some(t) if t != 0 => t,
                    _ => {
                        if !alive_set.contains(ppid) {
                            continue;
                        }
                        match process_creation_time(*ppid) {
                            Some(t) => {
                                sec.pid_creation.insert(*ppid, t);
                                t
                            }
                            None => continue,
                        }
                    }
                };
                let Some(child_creation) = process_creation_time(*pid) else {
                    // 子进程打不开（已在快照和打开之间退了，或权限不够），
                    // 这次不加，下次轮询如果还在、能打开再加，不会漏掉活进程。
                    continue;
                };
                let parent_alive = alive_set.contains(ppid);
                let parent_last_seen = sec.pid_last_seen_unix.get(ppid).copied();
                if !can_add_via_parent(parent_creation, parent_alive, parent_last_seen, child_creation) {
                    continue;
                }
                sec.pids.insert(*pid);
                sec.pid_creation.insert(*pid, child_creation);
                sec.pid_last_seen_unix.insert(*pid, now_unix);
                changed = true;
            }
        }
    }
}

fn check_sections_exited(shared: &Shared) {
    update_descendants(shared);
    let (alive_set, snap_list) = snapshot_processes();
    let parent_map: HashMap<u32, u32> = snap_list.into_iter().collect();
    let mut newly: Vec<String> = Vec::new();
    {
        let mut sections = match shared.sections.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        for sec in sections.iter_mut() {
            if sec.exited {
                continue;
            }
            let root_exited = if sec.root_handle == 0 {
                true
            } else {
                unsafe { WaitForSingleObject(sec.root_handle, 0) == 0 }
            };
            if !root_exited {
                continue;
            }
            // 退出判定以作业对象为准，再以锚点目录区分核心/外来：
            // 作业分配成功时，只看 1) 根 cmd 已退出；2) 作业对象里已无核心进程。
            // 快照后代不再决定退出，只留着做诊断。作业成员关系是内核自动继承的，
            // 搜狗 SGTool 这种从浏览器进程内部拉起来的共享助手也会进作业，
            // 所以只看作业空不空还不够：它的 exe 在 SogouInput 目录，不在
            // 锚点目录里，算外来进程，不阻塞退出、不被关闭/强杀。
            // 只有作业分配失败（job_handle 无效）时，才回退到快照判断，
            // 因为那时作业根本没有跟踪能力，快照是唯一的来源。
            let job_pids = query_job_pids(sec.job_handle);
            // 根退出时间每段只取一次，不在每个 pid 的 filter 里重复调。
            let root_exit = root_exit_time_of(sec);
            let job_core: Vec<u32> = job_pids
                .iter()
                .copied()
                .filter(|pid| {
                    let ppid = parent_map.get(pid).copied();
                    let path = cached_exe_path(sec, *pid);
                    is_core_process(sec, *pid, ppid, path.as_deref(), root_exit)
                })
                .collect();
            if !job_core.is_empty() {
                continue;
            }
            let job_valid = sec.job_handle != 0 && sec.job_handle != -1;
            if job_valid {
                sec.exited = true;
                newly.push(sec.name.clone());
                continue;
            }
            // 回退：作业无效时，快照后代要按 PID+创建时间比对：
            // 创建时间对不上的就是旧进程死了、新进程复用了 PID，
            // 直接从集合里清掉，不算活着。
            let mut reused: Vec<u32> = Vec::new();
            let any_desc_alive = sec
                .pids
                .iter()
                .filter(|pid| **pid != sec.root_pid)
                .any(|pid| {
                    if !alive_set.contains(pid) {
                        return false;
                    }
                    let recorded = sec.pid_creation.get(pid).copied().unwrap_or(0);
                    if recorded != 0 && !is_same_process(*pid, recorded) {
                        reused.push(*pid);
                        return false;
                    }
                    // 回退时同样按锚点区分核心/外来，外来进程不阻塞退出。
                    let ppid = parent_map.get(pid).copied();
                    let path = cached_exe_path(sec, *pid);
                    is_core_process(sec, *pid, ppid, path.as_deref(), root_exit)
                });
            for pid in reused {
                sec.pids.remove(&pid);
                sec.pid_creation.remove(&pid);
                sec.pid_last_seen_unix.remove(&pid);
            }
            if !any_desc_alive {
                sec.exited = true;
                newly.push(sec.name.clone());
            }
        }
    }
    for name in newly {
        log_msg(shared, &format!("退出: {}", name));
    }
    maybe_log_main_blockers(shared);
}

/// 主段根进程已退、后代还活着时写诊断日志：首次立即写，之后每 30 秒写一次。
/// 只在这种反常情况下写，正常使用（浏览器开着、根进程没退）不写。
fn maybe_log_main_blockers(shared: &Shared) {
    if shared.main_name.is_empty() {
        return;
    }
    let (main_name, root_exited, job_core, job_foreign, desc_alive, job_valid) = {
        let Ok(sections) = shared.sections.lock() else {
            return;
        };
        let Some(sec) = sections
            .iter()
            .find(|s| s.name.to_ascii_lowercase() == shared.main_name && !s.exited)
        else {
            return;
        };
        let job_valid = sec.job_handle != 0 && sec.job_handle != -1;
        let root_exited = if sec.root_handle == 0 {
            true
        } else {
            unsafe { WaitForSingleObject(sec.root_handle, 0) == 0 }
        };
        if !root_exited {
            return;
        }
        let job_pids = query_job_pids(sec.job_handle);
        let (alive_set, snap_list) = snapshot_processes();
        let parent_map: HashMap<u32, u32> = snap_list.into_iter().collect();
        let root_exit = root_exit_time_of(sec);
        let mut core = Vec::new();
        let mut foreign = Vec::new();
        for pid in &job_pids {
            let ppid = parent_map.get(pid).copied();
            let path = cached_exe_path(sec, *pid);
            if is_core_process(sec, *pid, ppid, path.as_deref(), root_exit) {
                core.push(*pid);
            } else {
                foreign.push(*pid);
            }
        }
        let mut desc: Vec<u32> = Vec::new();
        for pid in &sec.pids {
            if *pid == sec.root_pid {
                continue;
            }
            if !alive_set.contains(pid) {
                continue;
            }
            let recorded = sec.pid_creation.get(pid).copied().unwrap_or(0);
            if recorded != 0 && !is_same_process(*pid, recorded) {
                continue;
            }
            desc.push(*pid);
        }
        (sec.name.clone(), root_exited, core, foreign, desc, job_valid)
    };
    let _ = root_exited;
    // 只在核心进程还活着时报“段未判定退出”；只有外来进程时，
    // 主段按锚点规则已经退出了，不写这条日志，避免 SGTool 还活着时
    // 日志一直刷、好像还卡着的误导。外来进程单独列一行，方便排查。
    if job_valid {
        if job_core.is_empty() {
            return;
        }
    } else if job_core.is_empty() && desc_alive.is_empty() {
        return;
    }
    let should_log = {
        let Ok(mut last) = shared.diag_last_main_blocker.lock() else {
            return;
        };
        match *last {
            None => {
                *last = Some(Instant::now());
                true
            }
            Some(t) if t.elapsed() >= Duration::from_secs(30) => {
                *last = Some(Instant::now());
                true
            }
            Some(_) => false,
        }
    };
    if !should_log {
        return;
    }
    let names = snapshot_pid_names();
    let msg = if job_valid {
        format!(
            "主程序 {} 根进程已退出，但作业核心进程还活着，段未判定退出：作业核心进程 [{}]，已忽略的外来进程 [{}]，快照后代（仅诊断，不阻塞退出） [{}]",
            main_name,
            fmt_pid_list(&job_core, &names),
            fmt_pid_list(&job_foreign, &names),
            fmt_pid_list(&desc_alive, &names)
        )
    } else {
        format!(
            "主程序 {} 根进程已退出，但后代还活着，段未判定退出（作业无效，回退快照判断）：作业核心进程 [{}]，已忽略的外来进程 [{}]，后代活进程 [{}]",
            main_name,
            fmt_pid_list(&job_core, &names),
            fmt_pid_list(&job_foreign, &names),
            fmt_pid_list(&desc_alive, &names)
        )
    };
    log_diag(shared, &msg);
}

fn is_main_exited(shared: &Shared) -> bool {
    if shared.main_name.is_empty() {
        return false;
    }
    let sections = match shared.sections.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    sections
        .iter()
        .any(|s| s.name.to_ascii_lowercase() == shared.main_name && s.exited)
}

/// DogTray 拥有、可以关闭/强杀的进程集合：作业为准。
/// 作业有效时只含作业成员，快照-only 的 SGTool、v2rayN 之类共享/无关进程
/// 不在里面，退出时不会给它们发 WM_CLOSE、不会强杀它们。
/// 作业无效时回退到跟踪集合（快照），因为那时没有别的归属证据。
fn tracked_pids(shared: &Shared) -> HashSet<u32> {
    let mut set = HashSet::new();
    let (_, snap_list) = snapshot_processes();
    let parent_map: HashMap<u32, u32> = snap_list.into_iter().collect();
    if let Ok(sections) = shared.sections.lock() {
        for sec in sections.iter() {
            let root_exit = root_exit_time_of(sec);
            if sec.job_handle != 0 && sec.job_handle != -1 {
                for pid in query_job_pids(sec.job_handle) {
                    let ppid = parent_map.get(&pid).copied();
                    let path = cached_exe_path(sec, pid);
                    if is_core_process(sec, pid, ppid, path.as_deref(), root_exit) {
                        set.insert(pid);
                    }
                }
                // 根 cmd 活着时一定在作业里，上面已含；根已退时不在作业里，
                // 也不用再把它加回来（退出判定里根已退就不再等它）。
            } else {
                for pid in &sec.pids {
                    let ppid = parent_map.get(pid).copied();
                    let path = cached_exe_path(sec, *pid);
                    if is_core_process(sec, *pid, ppid, path.as_deref(), root_exit) {
                        set.insert(*pid);
                    }
                }
            }
        }
    }
    set
}

unsafe extern "system" fn enum_windows_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
    let ctx = &mut *(lparam as *mut EnumCtx);
    let mut pid: DWORD = 0;
    GetWindowThreadProcessId(hwnd, &mut pid);
    if (*ctx.tracked).contains(&pid) {
        (*ctx.out).push(hwnd);
    }
    1
}

fn enum_windows_for_pids(pids: &HashSet<u32>) -> Vec<HWND> {
    let mut hwnds: Vec<HWND> = Vec::new();
    let mut ctx = EnumCtx {
        tracked: pids,
        out: &mut hwnds,
    };
    unsafe {
        EnumWindows(Some(enum_windows_proc), &mut ctx as *mut EnumCtx as LPARAM);
    }
    hwnds
}

fn process_exe_path_from_handle(handle: HANDLE) -> Option<String> {
    if handle == 0 || handle == -1 {
        return None;
    }
    unsafe {
        let mut buf = vec![0u16; 1024];
        let mut size: DWORD = buf.len() as DWORD;
        let ok = QueryFullProcessImageNameW(handle, 0, buf.as_mut_ptr(), &mut size);
        if ok == 0 {
            return None;
        }
        Some(String::from_utf16_lossy(&buf[..size as usize]))
    }
}

/// 只按 exe 路径判断核心：用于句柄复核，不含直接子进程判断。
/// 路径拿不到时按核心算（保守，和 is_core_process 一致）；
/// 自动锚点目录为空时也按核心算（兜底，和 is_core_process 一致，
/// 手工锚点不参与这个空判断）。
fn is_core_by_path(sec: &Section, exe_path: Option<&str>) -> bool {
    let Some(path) = exe_path else {
        return true;
    };
    let lower = path.to_lowercase();
    if sec.anchor_files.contains(&lower) {
        return true;
    }
    if sec.anchor_dirs.iter().any(|d| lower.starts_with(d)) {
        return true;
    }
    if sec.manual_anchor_dirs.iter().any(|d| lower.starts_with(d)) {
        return true;
    }
    sec.anchor_dirs.is_empty()
}

/// 剩余进程列表：(PID, 所属段在 sections 里的下标, 判定当时记录的创建时间)。
/// 段下标和创建时间必须在这里一起返回，不能让调用方事后再按 PID 反查：
/// pids / pid_creation 里的旧条目基本不会被清理，A 段很早以前用过的 PID
/// 现在被 B 段的核心进程复用时，反查会拿到 A 段的旧时间和旧锚点，
/// 时间对不上、路径又不在 A 段锚点里，就会被判成外来而漏杀。
/// 这里返回的段下标就是判定核心时用的那一段，创建时间也是那一段里
/// 当时记录的那一个，和判定时的锚点一一对应。
fn remaining_pids(shared: &Shared) -> Vec<(u32, usize, u64)> {
    let (alive_set, snap_list) = snapshot_processes();
    let parent_map: HashMap<u32, u32> = snap_list.into_iter().collect();
    let mut remaining: Vec<(u32, usize, u64)> = Vec::new();
    if let Ok(sections) = shared.sections.lock() {
        for (sec_idx, sec) in sections.iter().enumerate() {
            let root_exit = root_exit_time_of(sec);
            if sec.job_handle != 0 && sec.job_handle != -1 {
                // 作业为准、锚点区分核心/外来：作业成员里只有核心进程才进
                // 关闭/强杀名单，外来进程（SGTool 等）即使在作业里也不杀。
                // 作业成员复用 PID 也是作业成员，创建时间比对不再用来排除核心进程。
                for pid in query_job_pids(sec.job_handle) {
                    if !alive_set.contains(&pid) {
                        continue;
                    }
                    let ppid = parent_map.get(&pid).copied();
                    let path = cached_exe_path(sec, pid);
                    if !is_core_process(sec, pid, ppid, path.as_deref(), root_exit) {
                        continue;
                    }
                    if pid == sec.root_pid {
                        let root_exited = if sec.root_handle == 0 {
                            true
                        } else {
                            unsafe { WaitForSingleObject(sec.root_handle, 0) == 0 }
                        };
                        if root_exited {
                            continue;
                        }
                    }
                    if !remaining.iter().any(|(p, _, _)| *p == pid) {
                        let recorded = sec.pid_creation.get(&pid).copied().unwrap_or(0);
                        remaining.push((pid, sec_idx, recorded));
                    }
                }
                continue;
            }
            // 回退：作业无效时用跟踪集合（快照），且强杀前比对创建时间：
            // 对不上就是 PID 被复用了，绝不杀它；外来进程同样不杀。
            for pid in &sec.pids {
                if !alive_set.contains(pid) {
                    continue;
                }
                let recorded = sec.pid_creation.get(pid).copied().unwrap_or(0);
                if recorded != 0 && !is_same_process(*pid, recorded) {
                    continue;
                }
                let ppid = parent_map.get(pid).copied();
                let path = cached_exe_path(sec, *pid);
                if !is_core_process(sec, *pid, ppid, path.as_deref(), root_exit) {
                    continue;
                }
                if *pid == sec.root_pid {
                    let root_exited = if sec.root_handle == 0 {
                        true
                    } else {
                        unsafe { WaitForSingleObject(sec.root_handle, 0) == 0 }
                    };
                    if root_exited {
                        continue;
                    }
                }
                if !remaining.iter().any(|(p, _, _)| *p == *pid) {
                    remaining.push((*pid, sec_idx, recorded));
                }
            }
        }
    }
    remaining
}

/// 强杀专用的剩余进程列表：判定核心时就持有进程句柄。
/// 原理：只要持有进程句柄，内核就不会回收这个进程对象、PID 也就不可能
/// 被别的进程复用，所以从判定到强杀这段时间内，句柄指向的永远是当初
/// 判定为核心的那个进程，不会变成刚好复用 PID 的 SGTool。
/// 句柄拿到后还要复核一次：快照和 OpenProcess 之间也可能发生复用，
/// 如果句柄的创建时间对不上当初记录的，就再看句柄进程的 exe 路径：
/// 路径仍在锚点里说明复用它的也是核心进程，句柄照杀无误；
/// 路径是外来目录说明复用它的是 SGTool 之类，关句柄、跳过不杀。
/// 句柄打不开时返回 0，强杀时走旧的按 PID 打开、创建时间校验的老路，
/// 不会因为打不开句柄就漏杀。
/// 调用方必须在强杀后关掉所有非 0 句柄，否则句柄泄漏、PID 一直占着。
fn remaining_pids_with_handles(shared: &Shared) -> Vec<(u32, HANDLE, u64)> {
    let entries = remaining_pids(shared);
    let mut out: Vec<(u32, HANDLE, u64)> = Vec::new();
    if entries.is_empty() {
        return out;
    }
    if let Ok(sections) = shared.sections.lock() {
        for (pid, sec_idx, recorded) in entries {
            if out.iter().any(|(p, _, _)| *p == pid) {
                continue;
            }
            let handle = unsafe {
                OpenProcess(
                    PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION,
                    0,
                    pid,
                )
            };
            if handle == 0 || handle == -1 {
                out.push((pid, 0, recorded));
                continue;
            }
            let mut keep = true;
            if recorded != 0 {
                let now = creation_time_from_handle(handle).unwrap_or(0);
                if now != recorded {
                    // 快照和打开之间发生了复用：按句柄进程自己的路径再判一次，
                    // 用 remaining_pids 返回的那个段的锚点，不按 PID 反查：
                    // 反查会拿到别的段很早以前用过这个 PID 的旧条目，
                    // 用旧段的锚点判新段的程序，会把该杀的判成外来而漏杀。
                    let path = process_exe_path_from_handle(handle);
                    let core = match sections.get(sec_idx) {
                        Some(sec) => is_core_by_path(sec, path.as_deref()),
                        None => true,
                    };
                    if !core {
                        log_msg(
                            shared,
                            &format!(
                                "跳过强杀 PID {}（句柄创建时间对不上、且路径是外来进程，疑似 PID 复用）",
                                pid
                            ),
                        );
                        keep = false;
                    }
                }
            }
            if keep {
                out.push((pid, handle, recorded));
            } else {
                unsafe {
                    CloseHandle(handle);
                }
            }
        }
    } else {
        // 拿不到段锁时无法复核，退回按 PID 的老路，句柄都记 0，
        // 但创建时间仍用 remaining_pids 返回的那一个，不反查。
        for (pid, _, recorded) in entries {
            out.push((pid, 0, recorded));
        }
    }
    out
}

fn wait_all_exited(shared: &Shared, timeout_ms: u64) -> bool {
    let mut waited: u64 = 0;
    while waited < timeout_ms {
        std::thread::sleep(Duration::from_millis(200));
        waited += 200;
        check_sections_exited(shared);
        let all_exited = match shared.sections.lock() {
            Ok(sections) => sections.iter().all(|s| s.exited),
            Err(_) => false,
        };
        if all_exited {
            return true;
        }
    }
    check_sections_exited(shared);
    match shared.sections.lock() {
        Ok(sections) => sections.iter().all(|s| s.exited),
        Err(_) => false,
    }
}

fn pids_of_windows(hwnds: &[HWND]) -> HashSet<u32> {
    let mut set = HashSet::new();
    for hwnd in hwnds {
        let mut pid: DWORD = 0;
        unsafe {
            GetWindowThreadProcessId(*hwnd, &mut pid);
        }
        if pid != 0 {
            set.insert(pid);
        }
    }
    set
}

fn send_shutdown(hwnds: &[HWND]) {
    // 模拟系统关机通知：先 WM_QUERYENDSESSION，再 WM_ENDSESSION
    // （wParam=TRUE 表示会话确实要结束，lParam=0 表示关机而非注销）。
    // 用 PostMessage，不会因某个窗口挂起而卡住主线程；
    // 收到关机通知的程序一般会自己保存数据并主动退出。
    for hwnd in hwnds {
        unsafe {
            PostMessageW(*hwnd, WM_QUERYENDSESSION, 0, 0);
            PostMessageW(*hwnd, WM_ENDSESSION, 1, 0);
        }
    }
}

fn force_kill_remaining(shared: &Shared) {
    let remaining = remaining_pids_with_handles(shared);
    if remaining.is_empty() {
        log_msg(shared, "所有子程序已退出");
        return;
    }
    log_msg(shared, &format!("优雅关闭后仍有 {} 个子程序未退出，强行终止", remaining.len()));
    // 作业成员集合：只用于句柄打不开时的老路。
    // 句柄能打开时，持有句柄本身就是归属证据，PID 复用不可能发生，
    // 不再需要“作业成员跳过创建时间校验”这条规则。
    let job_owned: HashSet<u32> = match shared.sections.lock() {
        Ok(sections) => sections
            .iter()
            .filter(|s| s.job_handle != 0 && s.job_handle != -1)
            .flat_map(|s| query_job_pids(s.job_handle))
            .collect(),
        Err(_) => HashSet::new(),
    };
    for (pid, handle, recorded) in &remaining {
        if *handle != 0 && *handle != -1 {
            // 句柄就是判定核心时拿到的那一个，直接杀它，绝不按 PID 重开。
            // 只要句柄还开着，PID 就不可能被复用，杀的永远是原进程。
            unsafe {
                TerminateProcess(*handle, 1);
                CloseHandle(*handle);
            }
            log_msg(shared, &format!("强行终止 PID {}", pid));
            continue;
        }
        if !job_owned.contains(pid) {
            // 句柄打不开时的老路：强杀前最后再比对一次创建时间，
            // 判断和强杀之间也可能发生复用。
            // 创建时间用 remaining_pids 返回的那一个，不按 PID 反查，
            // 否则会拿到别的段很早以前用过这个 PID 的旧条目。
            if *recorded != 0 && !is_same_process(*pid, *recorded) {
                log_msg(shared, &format!("跳过强杀 PID {}（创建时间对不上，疑似 PID 复用）", pid));
                continue;
            }
        }
        unsafe {
            let h = OpenProcess(PROCESS_TERMINATE, 0, *pid);
            if h != 0 && h != -1 {
                TerminateProcess(h, 1);
                CloseHandle(h);
            }
        }
        log_msg(shared, &format!("强行终止 PID {}", pid));
    }
    let pids: Vec<u32> = remaining.iter().map(|(pid, _, _)| *pid).collect();
    for _ in 0..10 {
        std::thread::sleep(Duration::from_millis(200));
        let (alive_set2, _) = snapshot_processes();
        if pids.iter().all(|pid| !alive_set2.contains(pid)) {
            break;
        }
    }
}

fn graceful_shutdown(shared: &Shared) {
    if shared.shutdown_done.load(Ordering::SeqCst) {
        return;
    }
    shared.shutdown.store(true, Ordering::SeqCst);
    log_msg(shared, "开始关闭子程序");
    // 给工作线程一点时间停下，不再启动下一个段。
    std::thread::sleep(Duration::from_millis(1000));
    update_descendants(shared);
    let tracked = tracked_pids(shared);
    if !tracked.is_empty() {
        // 统一流程，不按有无窗口分路、不按段串行，全局并行等：
        // 第 1 阶段：只给有窗口的程序发 WM_CLOSE，等 5 秒。
        //   无窗口的程序收不到 WM_CLOSE，但这 5 秒也要等，
        //   因为它可能依赖窗口程序——窗口程序退了，它才自行退出。
        // 第 2 阶段：重新枚举所有还没退出的程序（有窗口、没窗口都算），
        //   给它们发关机通知，再等 5 秒。
        //   重新枚举是关键：第 1 阶段后新弹出的确认框、隐藏窗口，
        //   以及原来无窗口、现在有了窗口的程序，都能收到。
        //   真正完全无窗口的程序，关机通知同样无窗口可收，
        //   这第 2 个 5 秒是给它因依赖程序退出而自行退出的时间。
        // 最后：还没退出的，全部强行终止。
        let alive_set: HashSet<u32> = remaining_pids(shared)
            .into_iter()
            .map(|(pid, _, _)| pid)
            .collect();
        let hwnds = enum_windows_for_pids(&alive_set);
        let windowed = pids_of_windows(&hwnds);
        let windowless_count = alive_set.iter().filter(|p| !windowed.contains(p)).count();
        for hwnd in &hwnds {
            unsafe {
                PostMessageW(*hwnd, WM_CLOSE, 0, 0);
            }
        }
        if windowless_count > 0 {
            log_msg(shared, &format!("第1阶段：已向 {} 个窗口发送关闭消息；{} 个无窗口进程一同等待（可能依赖窗口程序）", hwnds.len(), windowless_count));
        } else {
            log_msg(shared, &format!("第1阶段：已向 {} 个窗口发送关闭消息", hwnds.len()));
        }
        if wait_all_exited(shared, 5000) {
            log_msg(shared, "所有子程序已退出");
        } else {
            update_descendants(shared);
            let remaining_set: HashSet<u32> = remaining_pids(shared)
                .into_iter()
                .map(|(pid, _, _)| pid)
                .collect();
            let hwnds2 = enum_windows_for_pids(&remaining_set);
            send_shutdown(&hwnds2);
            let windowed2 = pids_of_windows(&hwnds2);
            let windowless2 = remaining_set.iter().filter(|p| !windowed2.contains(p)).count();
            if !hwnds2.is_empty() {
                log_msg(shared, &format!("第2阶段：向所有还没退出的程序发送关机通知，已向 {} 个窗口送出，{} 个无窗口进程一同等待", hwnds2.len(), windowless2));
            } else if windowless2 > 0 {
                log_msg(shared, &format!("第2阶段：所有还没退出的 {} 个进程均无窗口，关机通知无窗口可送，一同再等5秒", windowless2));
            }
            if wait_all_exited(shared, 5000) {
                log_msg(shared, "所有子程序已退出");
            } else {
                force_kill_remaining(shared);
            }
        }
    } else {
        log_msg(shared, "没有需要关闭的子程序");
    }
    {
        // 兜底清理：正常情况下临时文件在每段启动后 5 秒就已删掉，
        // 这里只删万一前面没删掉的残留（比如启动时正在关机、文件被锁住）。
        let temps = match shared.temp_bats.lock() {
            Ok(g) => g.clone(),
            Err(_) => Vec::new(),
        };
        for p in temps {
            let _ = std::fs::remove_file(&p);
        }
    }
    if let Ok(sections) = shared.sections.lock() {
        for sec in sections.iter() {
            if sec.root_handle != 0 {
                unsafe {
                    CloseHandle(sec.root_handle);
                }
            }
            // 作业句柄在杀完之后才关，关之前作业列表一直可用；
            // 这里不设 KILL_ON_JOB_CLOSE，所以关句柄本身不会再杀进程。
            if sec.job_handle != 0 && sec.job_handle != -1 {
                unsafe {
                    CloseHandle(sec.job_handle);
                }
            }
        }
    }
    log_msg(shared, "程序退出");
    shared.shutdown_done.store(true, Ordering::SeqCst);
}

/// 启动完成后删除临时批处理。
/// 删的时候 cmd 已经打开并读了这个文件（每段启动后都等了 5 秒），
/// 所以目录里不再出现它；如果 Windows 此时还锁着删不掉，
/// 就留在 shared.temp_bats 里，退出时再兜底删一次，不会一直残留。
fn delete_temp_bat(shared: &Shared, path: &PathBuf) -> bool {
    let ok = std::fs::remove_file(path).is_ok();
    if ok {
        if let Ok(mut temps) = shared.temp_bats.lock() {
            temps.retain(|p| p != path);
        }
        log_msg(shared, &format!("删除临时批处理: {}", path.display()));
    } else {
        log_msg(shared, &format!("临时批处理删除失败，退出时再删: {}", path.display()));
    }
    ok
}

fn launch_section(
    shared_arc: Arc<Shared>,
    name: &str,
    script: &str,
    current_dir: &PathBuf,
    manual_anchor_dirs: &[String],
) -> Option<PathBuf> {
    let shared: &Shared = &shared_arc;
    let dog_pid = std::process::id();
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let filename = format!("__dogtray_{}_{}.bat", dog_pid, sanitized);
    let temp_path = current_dir.join(&filename);
    let mut bat_text = script.to_string();
    if !bat_text.ends_with("\r\n") && !bat_text.ends_with('\n') {
        bat_text.push_str("\r\n");
    }
    let bat_bytes = wide_to_acp(&bat_text);
    if std::fs::write(&temp_path, &bat_bytes).is_err() {
        log_msg(shared, &format!("启动失败: {}（无法写入临时批处理）", name));
        return None;
    }
    if let Ok(mut temps) = shared.temp_bats.lock() {
        temps.push(temp_path.clone());
    }
    // cmd.exe 路径：系统目录，确定是 Windows 自带的 cmd。
    let cmd_path_str = unsafe {
        let mut buf = vec![0u16; 512];
        let len = GetSystemDirectoryW(buf.as_mut_ptr(), 512);
        if len > 0 && (len as usize) < buf.len() {
            let s = String::from_utf16_lossy(&buf[..len as usize]);
            format!("{}\\cmd.exe", s)
        } else {
            "cmd.exe".to_string()
        }
    };
    let cmd_path_w = wide(&cmd_path_str);
    let temp_str = temp_path.to_string_lossy().to_string();
    let cmdline_str = format!("/d /c \"{}\"", temp_str);
    let mut cmdline_w = wide(&cmdline_str);
    let cur_dir_w = wide(&current_dir.to_string_lossy());
    // 输出管道：stdout 和 stderr 写同一根管道，读端由后台线程读，
    // 子进程（包括 bat 里 start / 前台启动的程序）继承写端，
    // 所以它们的输出都会进这根管道，cmd 退了、后代还在时也不会提前 EOF。
    let (pipe_read, pipe_write, nul_handle, use_pipe) = unsafe {
        let mut sa = SECURITY_ATTRIBUTES {
            n_length: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lp_security_descriptor: null_mut(),
            b_inherit_handle: 1,
        };
        let mut r: HANDLE = 0;
        let mut w: HANDLE = 0;
        if CreatePipe(&mut r, &mut w, &sa, 0) == 0 {
            log_msg(shared, &format!("{}：创建输出管道失败，该段输出窗口将为空", name));
            (0, 0, 0, false)
        } else {
            // 读端不能被子进程继承，否则子进程拿着读端不放、管道清不掉。
            SetHandleInformation(r, HANDLE_FLAG_INHERIT, 0);
            sa.b_inherit_handle = 1;
            let nul_w = wide("NUL");
            let nul = CreateFileW(
                nul_w.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                &sa,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                0,
            );
            (r, w, if nul == -1 { 0 } else { nul }, true)
        }
    };
    unsafe {
        let mut si: STARTUPINFOW = zeroed();
        si.cb = size_of::<STARTUPINFOW>() as u32;
        if use_pipe {
            si.dw_flags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
            si.h_std_input = nul_handle;
            si.h_std_output = pipe_write;
            si.h_std_error = pipe_write;
        } else {
            si.dw_flags = STARTF_USESHOWWINDOW;
        }
        si.w_show_window = SW_HIDE as u16;
        let mut pi: PROCESS_INFORMATION = zeroed();
        // 关键：先以挂起方式创建 cmd，还没跑任何命令前就加入作业对象，
        // 再恢复线程。这样 cmd 及其所有后代从出生起就在作业里，
        // 中间进程活多久都不影响跟踪，作业列表永远能查到还活着的后代。
        // 作业对象不设任何限制（尤其不设 KILL_ON_JOB_CLOSE），
        // 只用来跟踪，不用来杀进程，杀进程仍走两阶段优雅关闭后强杀。
        let job_handle = CreateJobObjectW(null(), null());
        let ok = CreateProcessW(
            cmd_path_w.as_ptr(),
            cmdline_w.as_mut_ptr(),
            null(),
            null(),
            if use_pipe { 1 } else { 0 },
            CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED,
            null(),
            cur_dir_w.as_ptr(),
            &si,
            &mut pi,
        );
        if ok == 0 {
            log_msg(shared, &format!("启动失败: {}", name));
            if use_pipe {
                if pipe_write != 0 {
                    CloseHandle(pipe_write);
                }
                if pipe_read != 0 {
                    CloseHandle(pipe_read);
                }
                if nul_handle != 0 {
                    CloseHandle(nul_handle);
                }
            }
            if job_handle != 0 && job_handle != -1 {
                CloseHandle(job_handle);
            }
            // cmd 根本没起来，临时文件没人用，立刻删掉，不用等启动完成。
            let _ = std::fs::remove_file(&temp_path);
            if let Ok(mut temps) = shared.temp_bats.lock() {
                temps.retain(|p| p != &temp_path);
            }
            return None;
        }
        let mut job_handle = job_handle;
        if job_handle == 0 || job_handle == -1 {
            log_msg(shared, &format!("{}：创建作业对象失败，改用快照跟踪", name));
            job_handle = 0;
        } else if AssignProcessToJobObject(job_handle, pi.h_process) == 0 {
            log_msg(shared, &format!("{}：加入作业对象失败，改用快照跟踪", name));
            CloseHandle(job_handle);
            job_handle = 0;
        }
        if pi.h_thread != 0 {
            let prev = ResumeThread(pi.h_thread);
            if prev == u32::MAX {
                log_msg(shared, &format!("启动失败: {}（无法恢复线程）", name));
                TerminateProcess(pi.h_process, 1);
                CloseHandle(pi.h_process);
                CloseHandle(pi.h_thread);
                if use_pipe {
                    if pipe_write != 0 {
                        CloseHandle(pipe_write);
                    }
                    if pipe_read != 0 {
                        CloseHandle(pipe_read);
                    }
                    if nul_handle != 0 {
                        CloseHandle(nul_handle);
                    }
                }
                if job_handle != 0 {
                    CloseHandle(job_handle);
                }
                return None;
            }
            CloseHandle(pi.h_thread);
        }
        // 父进程关掉写端和 NUL，只留读端给读线程：
        // 这样写端只剩子进程拿着，子进程全退时读端才会读到 EOF。
        if use_pipe {
            if pipe_write != 0 {
                CloseHandle(pipe_write);
            }
            if nul_handle != 0 {
                CloseHandle(nul_handle);
            }
        }
        let mut pids = HashSet::new();
        pids.insert(pi.dw_process_id);
        let mut pid_creation = HashMap::new();
        // 根 cmd 的创建时间直接用它自己的句柄拿，最准，不用再打开一次。
        pid_creation.insert(
            pi.dw_process_id,
            creation_time_from_handle(pi.h_process).unwrap_or(0),
        );
        let mut pid_last_seen_unix = HashMap::new();
        pid_last_seen_unix.insert(pi.dw_process_id, unix_now());
        if let Ok(mut sections) = shared.sections.lock() {
            sections.push(Section {
                name: name.to_string(),
                root_pid: pi.dw_process_id,
                root_handle: pi.h_process,
                job_handle,
                pids,
                pid_creation,
                pid_last_seen_unix,
                // 手工锚点从段创建时就存在，单独存，不混进自动锚点：
                // 启动器活得久、真正的程序在别的目录时，真正的程序
                // 从一开始就是核心，不会被判成外来而提前退出。
                // 手工目录是用户显式配的，即使是盘符根也不降级，
                // 降级只针对自动记录的锚点（防止无意中记宽）。
                // 自动 anchor_dirs 仍从空开始，所以核心兜底、孤儿锚点
                // 的“自动锚点为空”判断不受手工锚点影响。
                anchor_dirs: HashSet::new(),
                anchor_files: HashSet::new(),
                manual_anchor_dirs: manual_anchor_dirs.iter().cloned().collect(),
                anchor_recorded: HashMap::new(),
                exited: false,
                output_text: String::new(),
                output_truncated: 0,
            });
        }
        if use_pipe && pipe_read != 0 {
            let reader_shared = shared_arc.clone();
            let reader_name = name.to_string();
            std::thread::spawn(move || {
                output_reader_thread(reader_shared, reader_name, pipe_read);
            });
        }
        log_msg(shared, &format!("启动: {} (PID {})", name, pi.dw_process_id));
        Some(temp_path)
    }
}

fn post_shutdown_to_main(shared: &Shared) {
    let hwnd = shared.hwnd.load(Ordering::SeqCst);
    if hwnd != 0 {
        unsafe {
            PostMessageW(hwnd, WM_APP_SHUTDOWN, 0, 0);
        }
    }
}

/// 等 5 秒，期间每 100 毫秒检查一次主程序是否退出。
/// 返回 true 表示要提前中止（正在关机，或主程序已退出并已通知主线程）。
fn wait_5_seconds_or_abort(shared: &Shared) -> bool {
    for _ in 0..50 {
        if shared.shutdown.load(Ordering::SeqCst) {
            return true;
        }
        check_sections_exited(shared);
        if is_main_exited(shared) {
            post_shutdown_to_main(shared);
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    if shared.shutdown.load(Ordering::SeqCst) {
        return true;
    }
    if is_main_exited(shared) {
        post_shutdown_to_main(shared);
        return true;
    }
    false
}

fn worker_thread(shared: Arc<Shared>, launches: Vec<(String, String, Vec<String>)>, current_dir: PathBuf) {
    for (idx, (name, script, sec_anchor_dirs)) in launches.iter().enumerate() {
        if shared.shutdown.load(Ordering::SeqCst) {
            break;
        }
        if script.trim().is_empty() {
            log_msg(&shared, &format!("跳过空程序: {}", name));
            continue;
        }
        // 手工锚点：全局（[main]）+ 本段，启动时就合并进段，
        // 这样启动器活得久、真正的程序在别的目录时，真正的程序从一开始就是核心，
        // 不用等自动锚点、也不依赖孤儿规则（孤儿只覆盖启动器早死的情况）。
        let mut manual: Vec<String> = shared.global_anchor_dirs.clone();
        for d in sec_anchor_dirs {
            if !manual.contains(d) {
                manual.push(d.clone());
            }
        }
        let temp = launch_section(shared.clone(), name, script, &current_dir, &manual);
        // 不管是不是最后一段，启动后都等 5 秒再删临时文件：
        // 这 5 秒足够 cmd 打开文件、读完脚本、把目标程序拉起来，
        // 又不会把临时文件留到 DogTray 退出（可能要跑很久）才删。
        // 最后一段也一样：等 5 秒、删掉，再进监控循环。
        let _ = idx;
        if wait_5_seconds_or_abort(&shared) {
            return;
        }
        if let Some(p) = temp {
            delete_temp_bat(&shared, &p);
        }
    }
    while !shared.shutdown.load(Ordering::SeqCst) {
        check_sections_exited(&shared);
        if is_main_exited(&shared) {
            post_shutdown_to_main(&shared);
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn to_display_text(s: &str) -> String {
    s.replace('\n', "\r\n")
}

fn output_window_at_bottom(edit_hwnd: HWND) -> bool {
    unsafe {
        let mut si = SCROLLINFO {
            cb_size: size_of::<SCROLLINFO>() as u32,
            f_mask: SIF_ALL,
            n_min: 0,
            n_max: 0,
            n_page: 0,
            n_pos: 0,
            n_track_pos: 0,
        };
        if GetScrollInfo(edit_hwnd, SB_VERT, &mut si) == 0 {
            return true;
        }
        if si.n_page == 0 {
            return true;
        }
        // 滚动条到底：当前位置 + 一页 >= 最大值。
        si.n_pos + si.n_page as i32 >= si.n_max
    }
}

fn open_output_window(shared: &Shared, section_name: &str) {
    // 已打开就提到前面，不重复开。
    if let Ok(windows) = shared.output_windows.lock() {
        for (hwnd, state) in windows.iter() {
            if state.section_name == section_name {
                unsafe {
                    if IsWindow(*hwnd) != 0 {
                        show_window(*hwnd);
                        return;
                    }
                }
            }
        }
    }
    unsafe {
        let h_instance = GetModuleHandleW(null());
        let class_name = wide("DogTrayOutputWindow");
        let title = wide(&format!("{} - 输出", section_name));
        let hwnd = CreateWindowExW(
            0,
            class_name.as_ptr(),
            title.as_ptr(),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            800,
            600,
            0,
            0,
            h_instance,
            null(),
        );
        if hwnd == 0 {
            return;
        }
        let edit_class = wide("EDIT");
        let edit_hwnd = CreateWindowExW(
            0,
            edit_class.as_ptr(),
            null(),
            WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_HSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_AUTOHSCROLL | ES_READONLY,
            0,
            0,
            800,
            600,
            hwnd,
            0,
            h_instance,
            null(),
        );
        if edit_hwnd != 0 {
            let font = GetStockObject(SYSTEM_FIXED_FONT);
            if font != 0 {
                SendMessageW(edit_hwnd, WM_SETFONT, font as WPARAM, 1);
            }
        }
        // 初始内容：内存里已有的全部输出（最多 2MB），打开就滚到底。
        let (initial_text, initial_len, initial_trunc) = if let Ok(sections) = shared.sections.lock() {
            if let Some(sec) = sections.iter().find(|s| s.name == section_name) {
                (sec.output_text.clone(), sec.output_text.len(), sec.output_truncated)
            } else {
                (String::new(), 0, 0)
            }
        } else {
            (String::new(), 0, 0)
        };
        if edit_hwnd != 0 && !initial_text.is_empty() {
            let disp = wide(&to_display_text(&initial_text));
            SetWindowTextW(edit_hwnd, disp.as_ptr());
        }
        if let Ok(mut windows) = shared.output_windows.lock() {
            windows.insert(
                hwnd,
                OutputWindowState {
                    section_name: section_name.to_string(),
                    edit_hwnd,
                    last_len: initial_len,
                    last_truncated: initial_trunc,
                },
            );
        }
        // 首次打开就按客户区尺寸放好 Edit，避免 WM_SIZE 在注册前发来、
        // 注册表里还查不到 Edit 句柄导致首次尺寸不对。
        {
            let mut rc: RECT = zeroed();
            if edit_hwnd != 0 && GetClientRect(hwnd, &mut rc) != 0 {
                MoveWindow(edit_hwnd, 0, 0, rc.right - rc.left, rc.bottom - rc.top, 1);
            }
        }
        SetTimer(hwnd, 1, 200, null());
        ShowWindow(hwnd, SW_SHOW);
        SetForegroundWindow(hwnd);
        if edit_hwnd != 0 {
            SendMessageW(edit_hwnd, WM_VSCROLL, SB_BOTTOM, 0);
        }
    }
}

fn update_output_window(shared: &Shared, hwnd: HWND) {
    let (section_name, edit_hwnd, last_len, last_trunc) = {
        let Ok(windows) = shared.output_windows.lock() else {
            return;
        };
        let Some(state) = windows.get(&hwnd) else {
            return;
        };
        (
            state.section_name.clone(),
            state.edit_hwnd,
            state.last_len,
            state.last_truncated,
        )
    };
    if edit_hwnd == 0 {
        return;
    }
    // 取当前缓存：截断过就全量刷新，否则只取新增后缀，避免 2MB 全量复制。
    let (full_text, suffix_text, cur_len, cur_trunc) = {
        let Ok(sections) = shared.sections.lock() else {
            return;
        };
        let Some(sec) = sections.iter().find(|s| s.name == section_name) else {
            return;
        };
        if sec.output_truncated != last_trunc
            || last_len > sec.output_text.len()
            || !sec.output_text.is_char_boundary(last_len)
        {
            // 截断、长度回缩或 last_len 不在字符边界时都全量刷新，绝不按字节切片。
            (Some(sec.output_text.clone()), None, sec.output_text.len(), sec.output_truncated)
        } else if sec.output_text.len() > last_len {
            (
                None,
                Some(sec.output_text[last_len..].to_string()),
                sec.output_text.len(),
                sec.output_truncated,
            )
        } else {
            return;
        }
    };
    unsafe {
        if let Some(full) = full_text {
            // 全量刷新（2MB 截断后）也尊重跟随规则：之前到底才滚到底，
            // 否则不强制到底（Edit 全量替换后默认在顶部，用户可再滚）。
            let was_bottom = output_window_at_bottom(edit_hwnd);
            let disp = wide(&to_display_text(&full));
            SetWindowTextW(edit_hwnd, disp.as_ptr());
            if was_bottom {
                SendMessageW(edit_hwnd, WM_VSCROLL, SB_BOTTOM, 0);
            }
        } else if let Some(suffix) = suffix_text {
            if suffix.is_empty() {
                return;
            }
            // 更新前先判断是否到底：到底才跟随，手动滚离则恢复原位置。
            let at_bottom = output_window_at_bottom(edit_hwnd);
            let first_visible_before = SendMessageW(edit_hwnd, EM_GETFIRSTVISIBLELINE, 0, 0);
            SendMessageW(edit_hwnd, EM_SETSEL, usize::MAX, -1);
            let disp = wide(&to_display_text(&suffix));
            SendMessageW(edit_hwnd, EM_REPLACESEL, 0, disp.as_ptr() as LPARAM);
            if at_bottom {
                SendMessageW(edit_hwnd, WM_VSCROLL, SB_BOTTOM, 0);
            } else {
                let first_visible_after = SendMessageW(edit_hwnd, EM_GETFIRSTVISIBLELINE, 0, 0);
                let delta = first_visible_before - first_visible_after;
                if delta != 0 {
                    SendMessageW(edit_hwnd, EM_LINESCROLL, 0, delta);
                }
            }
        }
    }
    if let Ok(mut windows) = shared.output_windows.lock() {
        if let Some(state) = windows.get_mut(&hwnd) {
            state.last_len = cur_len;
            state.last_truncated = cur_trunc;
        }
    }
}

unsafe extern "system" fn output_wnd_proc(hwnd: HWND, msg: UINT, w_param: WPARAM, l_param: LPARAM) -> LRESULT {
    match msg {
        WM_SIZE => {
            if let Some(shared) = SHARED.get() {
                let edit_hwnd = if let Ok(windows) = shared.output_windows.lock() {
                    windows.get(&hwnd).map(|s| s.edit_hwnd).unwrap_or(0)
                } else {
                    0
                };
                if edit_hwnd != 0 {
                    let mut rc: RECT = zeroed();
                    if GetClientRect(hwnd, &mut rc) != 0 {
                        MoveWindow(edit_hwnd, 0, 0, rc.right - rc.left, rc.bottom - rc.top, 1);
                    }
                }
            }
            0
        }
        WM_TIMER => {
            if w_param == 1 {
                if let Some(shared) = SHARED.get() {
                    update_output_window(shared, hwnd);
                }
            }
            0
        }
        WM_CLOSE => {
            DestroyWindow(hwnd);
            0
        }
        WM_DESTROY => {
            KillTimer(hwnd, 1);
            if let Some(shared) = SHARED.get() {
                if let Ok(mut windows) = shared.output_windows.lock() {
                    windows.remove(&hwnd);
                }
            }
            0
        }
        _ => DefWindowProcW(hwnd, msg, w_param, l_param),
    }
}

// DestroyWindow 在 user32 里，上面 output_wnd_proc 用到，单独声明避免改动原有 extern 块顺序。
#[link(name = "user32")]
extern "system" {
    fn DestroyWindow(h_wnd: HWND) -> BOOL;
}

unsafe fn tray_data(hwnd: HWND, icon: HICON, tip: &str) -> NOTIFYICONDATAW {
    let mut data: NOTIFYICONDATAW = zeroed();
    data.cb_size = size_of::<NOTIFYICONDATAW>() as u32;
    data.h_wnd = hwnd;
    data.u_id = 1;
    data.u_flags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
    data.u_callback_message = WM_TRAY;
    data.h_icon = icon;
    let tip_w = wide(tip);
    let n = tip_w.len().min(127);
    // wide() 自带结尾 0，截到 127 个单位内，保证 szTip 以 0 结尾。
    if n > 0 {
        data.sz_tip[..n].copy_from_slice(&tip_w[..n]);
    }
    data
}

unsafe fn remove_tray(hwnd: HWND) {
    let mut data = tray_data(hwnd, 0, "");
    Shell_NotifyIconW(NIM_DELETE, &mut data);
}

unsafe fn show_window(hwnd: HWND) {
    if IsIconic(hwnd) != 0 {
        ShowWindow(hwnd, SW_RESTORE);
    } else {
        ShowWindow(hwnd, SW_SHOW);
    }
    SetForegroundWindow(hwnd);
}

unsafe fn show_exit_menu(hwnd: HWND) {
    let menu = CreatePopupMenu();
    if menu == 0 {
        return;
    }
    let text = wide("退出程序");
    AppendMenuW(menu, MF_STRING, ID_EXIT as usize, text.as_ptr());
    let mut pt: POINT = zeroed();
    if GetCursorPos(&mut pt) == 0 {
        pt.x = 0;
        pt.y = 0;
    }
    SetForegroundWindow(hwnd);
    let chosen = TrackPopupMenu(menu, TPM_RIGHTBUTTON | TPM_RETURNCMD, pt.x, pt.y, 0, hwnd, null());
    PostMessageW(hwnd, 0, 0, 0);
    DestroyMenu(menu);
    if chosen as u32 == ID_EXIT {
        if let Some(shared) = SHARED.get() {
            graceful_shutdown(shared);
        }
        remove_tray(hwnd);
        PostQuitMessage(0);
    } else if let Some(shared) = SHARED.get() {
        // 菜单内部消息循环吞掉退出消息的口子：
        // 菜单打开期间，工作线程若判定主程序退出，会发 WM_APP_SHUTDOWN，
        // 它会被菜单内部循环先收到并处理（关机已跑、子程序已杀、已发退出消息），
        // 但那个退出消息会被菜单内部循环消耗掉，菜单按取消返回，
        // 外层若不再补发，主循环就会一直等消息，DogTray 活着、子程序已死。
        // 所以菜单返回后不管结果如何，只要关机已经跑过了，就在外层再补发一次。
        if shared.shutdown_done.load(Ordering::SeqCst) {
            remove_tray(hwnd);
            PostQuitMessage(0);
        }
    }
}

unsafe extern "system" fn wnd_proc(hwnd: HWND, msg: UINT, w_param: WPARAM, l_param: LPARAM) -> LRESULT {
    match msg {
        WM_TRAY => {
            match l_param as u32 {
                WM_LBUTTONUP | WM_LBUTTONDBLCLK => show_window(hwnd),
                WM_RBUTTONUP => show_exit_menu(hwnd),
                _ => {}
            }
            0
        }
        WM_APP_SHUTDOWN => {
            if let Some(shared) = SHARED.get() {
                graceful_shutdown(shared);
            }
            remove_tray(hwnd);
            PostQuitMessage(0);
            0
        }
        WM_COMMAND => {
            let cmd_id = (w_param & 0xFFFF) as u32;
            if cmd_id == ID_EXIT {
                if let Some(shared) = SHARED.get() {
                    graceful_shutdown(shared);
                }
                remove_tray(hwnd);
                PostQuitMessage(0);
            } else if cmd_id >= BUTTON_ID_BASE {
                // 主界面按钮：每个 [XXX] 一个，ID = 2000 + 段序号。
                if let Some(shared) = SHARED.get() {
                    let idx = (cmd_id - BUTTON_ID_BASE) as usize;
                    if let Some(name) = shared.button_sections.get(idx).cloned() {
                        open_output_window(shared, &name);
                    }
                }
            }
            0
        }
        WM_CLOSE => {
            ShowWindow(hwnd, SW_HIDE);
            0
        }
        WM_DESTROY => {
            remove_tray(hwnd);
            PostQuitMessage(0);
            0
        }
        _ => DefWindowProcW(hwnd, msg, w_param, l_param),
    }
}

fn main() {
    unsafe {
        let h_instance = GetModuleHandleW(null());
        if h_instance == 0 {
            return;
        }
        let stem = exe_stem();
        let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let config_path = current_dir.join("start_config.ini");
        let config_bytes = match std::fs::read(&config_path) {
            Ok(b) => b,
            Err(_) => {
                show_missing_config(&stem);
                return;
            }
        };
        let config_text = decode_bytes(&config_bytes);
        let cfg = parse_config(&config_text);
        let tray_tip = cfg.title.clone().unwrap_or_else(|| stem.clone());
        let log_path = current_dir.join(format!("{}.log", stem));
        let button_sections: Vec<String> = cfg.sections.iter().map(|(n, _, _)| n.clone()).collect();
        let shared = Arc::new(Shared {
            sections: Mutex::new(Vec::new()),
            shutdown: AtomicBool::new(false),
            shutdown_done: AtomicBool::new(false),
            log_enabled: cfg.output_log,
            log_path,
            main_name: cfg.main_name.clone(),
            hwnd: AtomicIsize::new(0),
            temp_bats: Mutex::new(Vec::new()),
            exe_stem: stem.clone(),
            tray_tip: tray_tip.clone(),
            button_sections,
            global_anchor_dirs: cfg.global_anchor_dirs.clone(),
            output_windows: Mutex::new(HashMap::new()),
            diag_last_main_blocker: Mutex::new(None),
        });
        if shared.log_enabled {
            log_msg(&shared, "程序启动");
        }
        let _ = SHARED.set(shared.clone());

        let class_name = wide("DogTrayWindow");
        let title = wide(&tray_tip);
        let mut icon = LoadIconW(h_instance, 1 as *const u16);
        if icon == 0 {
            icon = LoadIconW(0, IDI_APPLICATION as *const u16);
        }
        let cursor = LoadCursorW(0, IDC_ARROW as *const u16);
        let wc = WNDCLASSEXW {
            cb_size: size_of::<WNDCLASSEXW>() as u32,
            style: 0,
            lpfn_wnd_proc: Some(wnd_proc),
            cb_cls_extra: 0,
            cb_wnd_extra: 0,
            h_instance,
            h_icon: icon,
            h_cursor: cursor,
            hbr_background: (COLOR_WINDOW + 1) as HBRUSH,
            lpsz_menu_name: null(),
            lpsz_class_name: class_name.as_ptr(),
            h_icon_sm: icon,
        };
        if RegisterClassExW(&wc) == 0 {
            return;
        }
        // 输出窗口类：和主界面分开，关输出窗口只销毁它自己，不退出 DogTray。
        let out_class_name = wide("DogTrayOutputWindow");
        let wc_out = WNDCLASSEXW {
            cb_size: size_of::<WNDCLASSEXW>() as u32,
            style: 0,
            lpfn_wnd_proc: Some(output_wnd_proc),
            cb_cls_extra: 0,
            cb_wnd_extra: 0,
            h_instance,
            h_icon: icon,
            h_cursor: cursor,
            hbr_background: (COLOR_WINDOW + 1) as HBRUSH,
            lpsz_menu_name: null(),
            lpsz_class_name: out_class_name.as_ptr(),
            h_icon_sm: icon,
        };
        let _ = RegisterClassExW(&wc_out);
        let hwnd = CreateWindowExW(
            0,
            class_name.as_ptr(),
            title.as_ptr(),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            640,
            480,
            0,
            0,
            h_instance,
            null(),
        );
        if hwnd == 0 {
            return;
        }
        shared.hwnd.store(hwnd, Ordering::SeqCst);
        // 主界面：顶部一行说明，下面每个 [XXX] 一个按钮（不含 [main]）。
        {
            let label_class = wide("STATIC");
            let label_text = wide("点击按钮查看对应程序的输出");
            let _ = CreateWindowExW(
                0,
                label_class.as_ptr(),
                label_text.as_ptr(),
                WS_CHILD | WS_VISIBLE,
                20,
                10,
                600,
                20,
                hwnd,
                1999,
                h_instance,
                null(),
            );
            for (i, (name, _, _)) in cfg.sections.iter().enumerate() {
                let btn_class = wide("BUTTON");
                let btn_text = wide(name);
                let _ = CreateWindowExW(
                    0,
                    btn_class.as_ptr(),
                    btn_text.as_ptr(),
                    WS_CHILD | WS_VISIBLE,
                    20,
                    40 + (i as i32) * 50,
                    600,
                    40,
                    hwnd,
                    (BUTTON_ID_BASE as usize + i) as HMENU,
                    h_instance,
                    null(),
                );
            }
        }
        let mut data = tray_data(hwnd, icon, &tray_tip);
        Shell_NotifyIconW(NIM_ADD, &mut data);

        // 启动器在后台线程跑，主线程留给托盘消息循环。
        let worker_shared = shared.clone();
        let launches = cfg.sections.clone();
        let worker_dir = current_dir.clone();
        std::thread::spawn(move || {
            worker_thread(worker_shared, launches, worker_dir);
        });

        let mut msg: MSG = zeroed();
        while GetMessageW(&mut msg, 0, 0, 0) > 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if !shared.shutdown_done.load(Ordering::SeqCst) {
            graceful_shutdown(&shared);
        }
        remove_tray(hwnd);
        let _ = shared.exe_stem;
        let _ = shared.tray_tip;
    }
}
