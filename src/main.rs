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
use std::time::Duration;

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
    exited: bool,
    output_text: String,
    output_truncated: u64,
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
    output_windows: Mutex<HashMap<HWND, OutputWindowState>>,
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
    fn CreateToolhelp32Snapshot(dw_flags: DWORD, th32_process_id: DWORD) -> HANDLE;
    fn Process32FirstW(h_snapshot: HANDLE, lppe: *mut PROCESSENTRY32W) -> BOOL;
    fn Process32NextW(h_snapshot: HANDLE, lppe: *mut PROCESSENTRY32W) -> BOOL;
    fn CreateJobObjectW(lp_job_attributes: *const c_void, lp_name: *const u16) -> HANDLE;
    fn AssignProcessToJobObject(h_job: HANDLE, h_process: HANDLE) -> BOOL;
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
        let drain_to = if let Some(pos) = sec.output_text[excess..].find('\n') {
            excess + pos + 1
        } else {
            let mut s = excess;
            while s < sec.output_text.len() && !sec.output_text.is_char_boundary(s) {
                s += 1;
            }
            s
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
    sections: Vec<(String, String)>,
    main_name: String,
    output_log: bool,
    title: Option<String>,
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
    let mut sections: Vec<(String, String)> = Vec::new();
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
                        _ => {}
                    }
                }
            }
        } else {
            let script = lines.join("\r\n");
            sections.push((name, script));
        }
    }
    Config {
        sections,
        main_name,
        output_log,
        title,
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
    let (_set, list) = snapshot_processes();
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
    for (idx, sec) in sections.iter_mut().enumerate() {
        // 作业对象是内核级跟踪：中间进程活多久都不影响，
        // 它的子进程一创建就自动进作业，查作业列表就能直接拿到。
        if idx < job_lists.len() {
            for pid in &job_lists[idx] {
                sec.pids.insert(*pid);
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for (pid, ppid) in &list {
                if sec.pids.contains(ppid) && !sec.pids.contains(pid) {
                    sec.pids.insert(*pid);
                    changed = true;
                }
            }
        }
    }
}

fn check_sections_exited(shared: &Shared) {
    update_descendants(shared);
    let (alive_set, _) = snapshot_processes();
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
            // 只有三者同时满足才算整段退出：
            // 1) 根 cmd 已退出；2) 作业对象里已无活进程；
            // 3) 快照里也没有该段的后代活着（兜底脱离作业的进程）。
            // 以前只看 1) 和 3)，Chrome 的 launcher 死得太快、
            // 浏览器没进跟踪集合，就会被误判为已退出。
            let job_pids = query_job_pids(sec.job_handle);
            if !job_pids.is_empty() {
                continue;
            }
            let any_desc_alive = sec
                .pids
                .iter()
                .filter(|pid| **pid != sec.root_pid)
                .any(|pid| alive_set.contains(pid));
            if !any_desc_alive {
                sec.exited = true;
                newly.push(sec.name.clone());
            }
        }
    }
    for name in newly {
        log_msg(shared, &format!("退出: {}", name));
    }
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

fn tracked_pids(shared: &Shared) -> HashSet<u32> {
    let mut set = HashSet::new();
    if let Ok(sections) = shared.sections.lock() {
        for sec in sections.iter() {
            for pid in &sec.pids {
                set.insert(*pid);
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

fn remaining_pids(shared: &Shared) -> Vec<u32> {
    let (alive_set, _) = snapshot_processes();
    let mut remaining: Vec<u32> = Vec::new();
    if let Ok(sections) = shared.sections.lock() {
        for sec in sections.iter() {
            for pid in &sec.pids {
                if !alive_set.contains(pid) {
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
                if !remaining.contains(pid) {
                    remaining.push(*pid);
                }
            }
        }
    }
    remaining
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
    let remaining = remaining_pids(shared);
    if remaining.is_empty() {
        log_msg(shared, "所有子程序已退出");
        return;
    }
    log_msg(shared, &format!("优雅关闭后仍有 {} 个子程序未退出，强行终止", remaining.len()));
    for pid in &remaining {
        unsafe {
            let h = OpenProcess(PROCESS_TERMINATE, 0, *pid);
            if h != 0 && h != -1 {
                TerminateProcess(h, 1);
                CloseHandle(h);
            }
        }
        log_msg(shared, &format!("强行终止 PID {}", pid));
    }
    for _ in 0..10 {
        std::thread::sleep(Duration::from_millis(200));
        let (alive_set2, _) = snapshot_processes();
        if remaining.iter().all(|pid| !alive_set2.contains(pid)) {
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
        let alive_set: HashSet<u32> = remaining_pids(shared).into_iter().collect();
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
            let remaining_set: HashSet<u32> = remaining_pids(shared).into_iter().collect();
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

fn launch_section(shared_arc: Arc<Shared>, name: &str, script: &str, current_dir: &PathBuf) -> Option<PathBuf> {
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
        if let Ok(mut sections) = shared.sections.lock() {
            sections.push(Section {
                name: name.to_string(),
                root_pid: pi.dw_process_id,
                root_handle: pi.h_process,
                job_handle,
                pids,
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

fn worker_thread(shared: Arc<Shared>, launches: Vec<(String, String)>, current_dir: PathBuf) {
    for (idx, (name, script)) in launches.iter().enumerate() {
        if shared.shutdown.load(Ordering::SeqCst) {
            break;
        }
        if script.trim().is_empty() {
            log_msg(&shared, &format!("跳过空程序: {}", name));
            continue;
        }
        let temp = launch_section(shared.clone(), name, script, &current_dir);
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
        if sec.output_truncated != last_trunc {
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
        let button_sections: Vec<String> = cfg.sections.iter().map(|(n, _)| n.clone()).collect();
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
            output_windows: Mutex::new(HashMap::new()),
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
            for (i, (name, _)) in cfg.sections.iter().enumerate() {
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
