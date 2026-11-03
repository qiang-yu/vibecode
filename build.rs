use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let rc = manifest.join("resource.rc");
    println!("cargo:rerun-if-changed=resource.rc");
    println!("cargo:rerun-if-changed=dog.ico");

    if target.contains("windows-gnu") {
        // Linux 交叉编译：x86_64-w64-mingw32-windres；
        // Windows 本地 MinGW：windres.exe。
        let obj = out_dir.join("resource.o");
        for tool in ["x86_64-w64-mingw32-windres", "windres", "windres.exe"] {
            match Command::new(tool)
                .arg(&rc)
                .arg("-O")
                .arg("coff")
                .arg("-o")
                .arg(&obj)
                .status()
            {
                Ok(status) if status.success() => {
                    println!("cargo:rustc-link-arg={}", obj.display());
                    return;
                }
                _ => continue,
            }
        }
        println!("cargo:warning=windres not found, building without embedded icon");
    } else if target.contains("windows-msvc") {
        // Windows 本地 MSVC：用 Windows SDK 的 rc.exe 编 resource.rc。
        // 需要在“开发者命令提示符”里编译，rc.exe 才在 PATH 里。
        // 找不到 rc.exe 也不中断编译，只是 exe 图标会回退为系统默认图标。
        let res = out_dir.join("resource.res");
        for tool in ["rc.exe", "rc"] {
            match Command::new(tool)
                .arg("/fo")
                .arg(&res)
                .arg(&rc)
                .status()
            {
                Ok(status) if status.success() => {
                    println!("cargo:rustc-link-arg={}", res.display());
                    return;
                }
                _ => continue,
            }
        }
        println!("cargo:warning=rc.exe not found, building without embedded icon");
    }
}
