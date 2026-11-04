# DogTray

Windows 托盘启动器，纯 Rust + Windows 自身 API，没有第三方 crate 依赖。

- 启动时读当前目录下的 `start_config.ini`；找不到就弹对话框
  “缺少配置文件 start_config.ini”，点确定后退出。
- 配置按 `[XXXX]` 分段，每段正文就是一段 Windows cmd 批处理，
  按段顺序启动，每段启动后等 5 秒再启动下一段，不弹黑窗口。
- `[main]` 里配主程序和其它选项，例如：

```ini
[xray]
cd frpSchool-win
run.bat

[chrome]
set CurrentDir=%~dp0
"App\Chrome.exe" --start-maximized --proxy-server="127.0.0.1:11083"

[main]
main=chrome
output_log=false
title="海大穿梭"
```

- `main=chrome`：chrome 段是主程序。主程序及其全部子进程都退出后，
  DogTray 关掉其它段再退出；DogTray 自己退出时也关掉全部子程序。
- `output_log=true`：在当前目录生成和程序同名的 `.log`
  （如 `DogTray.log`），记时间、哪个程序启动、哪个程序退出。
- `title="..."`：托盘悬停文字；没配就用 exe 文件名（不带 `.exe`）。
- 托盘左键打开主界面：每个 `[XXX]` 一个按钮（`[main]` 不算）。
  点按钮打开该程序的输出窗口，显示它的 stdout + stderr。
  每段输出在内存里只留最近 2MB，超了删前面、新信息追加到后面。
  输出窗口每 200ms 刷新：滚动条到底才跟随滚动，手动滚离则不跟随；
  关闭输出窗口不杀程序，再点主界面按钮可重新打开。
- 退出时统一优雅关闭：先给有窗口的程序发 `WM_CLOSE` 等 5 秒（无窗口的程序一同等，因为它可能依赖窗口程序）；然后重新枚举所有还没退出的程序，给它们发关机通知（`WM_QUERYENDSESSION` + `WM_ENDSESSION`）再等 5 秒；还不退的才全部强行终止。托盘菜单返回后若关机已跑过，外层会补发退出消息，避免菜单内部循环吞掉退出消息导致 DogTray 本身不退。子进程按 PID + 创建时间跟踪，防止 PID 复用后误判活着或杀错进程。主段根进程已退、后代还活着时，每 30 秒写一条诊断日志（进程 PID + 进程名），即使 output_log=false 也写，只在这种反常情况下写。
  子程序用作业对象（Job Object）+ 进程快照双重跟踪，短命的中间
  launcher 退了、真正的浏览器还在时也不会误判为整段已退出。
  快照父子链加入时必须通过创建时间顺序检查：子进程创建时间晚于父进程，
  且父进程已死时子进程创建时间不能晚于父进程最后一次活着 + 5 秒，
  否则不加入，防止死父进程 PID 被复用后把 v2rayN 之类的无关进程树误吸进来。
  退出判定和强杀都以作业对象为准：作业分配成功时，根进程已退且作业空
  就算整段退出，快照后代（比如搜狗输入法 SGTool 这种按需拉起、父进程指向
  浏览器、但不在作业里的共享助手）不再阻塞退出，也不会被发关闭消息或强杀；
  只有作业分配失败时才回退到快照判断。
- 运行期间每段会生成一个 `__dogtray_*.bat` 临时文件（为了让脚本里的
  `%~dp0`、相对路径按原意工作），每段启动后 5 秒就删掉，不等到退出。

## 目录结构

```
Cargo.toml          包配置（无第三方依赖）
Cargo.lock
build.rs            编译时把 dog.ico 嵌进 exe（windres / rc.exe）
resource.rc         图标资源：1 ICON "dog.ico"
dog.ico             卡通狗头图标（256/48/32/16）
generate_icon.py    重新生成 dog.ico 的脚本（需要 Pillow）
src/main.rs         全部程序逻辑（托盘、启动、跟踪、关闭、输出窗口）
.cargo/config.toml  Linux 交叉编译时用的 mingw 链接器配置
```

## 编译（推荐：Linux 交叉编译，和作者原来一样）

```bash
rustup target add x86_64-pc-windows-gnu
# Ubuntu/Debian：
sudo apt install gcc-mingw-w64-x86-64
cargo build --release --target x86_64-pc-windows-gnu
# 产物：target/x86_64-pc-windows-gnu/release/dog-tray.exe
# 改名成 DogTray.exe，和 start_config.ini 放同一个文件夹里运行。
```

`.cargo/config.toml` 里已经配好了
`linker = "x86_64-w64-mingw32-gcc"`，不用再手动传链接器。

## 编译（Windows 本地）

先装 Rust（https://rustup.rs/），然后二选一：

1. MSVC（推荐在“开发者命令提示符”里，rc.exe 才在 PATH 里）：

```bat
cargo build --release
```

2. MinGW（MSYS2 的 mingw64 环境）：

```bat
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu
```

如果找不到 `windres` / `rc.exe`，编译不会中断，只是 exe 图标会
回退为系统默认图标，程序功能不受影响。

## 运行

把 exe 和 `start_config.ini` 放同一个文件夹，双击 exe 即可。
注意：例子文件如果叫 `start_config-2.ini`，要改名成 `start_config.ini`
程序才会读到。右下角托盘图标是卡通狗头；左键打开主界面，
右键菜单“退出程序”退出；主界面点 X 只隐藏回托盘，不退出。
