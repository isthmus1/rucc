//! `__bf16`, the bfloat16 type, which gcc names before any header on x86 and AArch64. gcc's own
//! `<immintrin.h>` declares the AVX-512 BF16 intrinsics with it, so without it nothing that
//! includes that header can be read on x86-64. The facts are measured from gcc 16.2.1 on x86-64
//! Linux: `sizeof` and `_Alignof` are two, the usual arithmetic conversions rank it by precision
//! (so `__bf16 + _Float16` is a `_Float16`), and the name may be shadowed because it is not a
//! keyword.

use std::io::Write as _;
use std::process::{Command, Output, Stdio};

/// rucc's output for `source` on stdin, checked and nothing more.
fn check(target: &str, source: &str) -> Output {
    run(target, &["-fsyntax-only"], source)
}

/// rucc's output for `source` on stdin with `args` before it.
fn run(target: &str, args: &[&str], source: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rucc"))
        .env("LC_ALL", "C")
        .arg(format!("--target={target}"))
        .arg("-std=gnu23")
        .args(args)
        .args(["-x", "c", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the compiler is built before its own tests run");
    child.stdin.take().expect("a pipe for the input").write_all(source.as_bytes()).unwrap();
    child.wait_with_output().expect("the compiler finished")
}

fn accepted(target: &str, source: &str) {
    let out = check(target, source);
    assert!(out.status.success(), "{target}: {}", String::from_utf8_lossy(&out.stderr));
}

const TYPE_AND_RANK: &str = "\
__bf16 b;
_Float16 h;
float f;
__bf16 widen(float);
_Static_assert(sizeof(__bf16) == 2 && _Alignof(__bf16) == 2, \"two bytes\");
_Static_assert(_Generic(b + b, __bf16: 1, default: 0), \"__bf16 + __bf16\");
_Static_assert(_Generic(b + h, _Float16: 1, default: 0), \"__bf16 + _Float16\");
_Static_assert(_Generic(h + b, _Float16: 1, default: 0), \"_Float16 + __bf16\");
_Static_assert(_Generic(b + f, float: 1, default: 0), \"__bf16 + float\");
_Static_assert(_Generic(b + 1, __bf16: 1, default: 0), \"__bf16 + int\");
_Static_assert(_Generic(b, _Float16: 0, float: 0, __bf16: 1), \"a type of its own\");
";

#[test]
fn bfloat16_is_a_type_of_its_own_on_x86_64() {
    accepted("x86_64-linux-gnu", TYPE_AND_RANK);
}

#[test]
fn bfloat16_is_a_type_on_i386_and_aarch64_too() {
    accepted("i686-linux-gnu", "__bf16 b; _Static_assert(sizeof b == 2, \"two bytes\");");
    accepted("aarch64-linux-gnu", "__bf16 b; _Static_assert(sizeof b == 2, \"two bytes\");");
}

#[test]
fn the_name_is_not_a_keyword() {
    accepted("x86_64-linux-gnu", "int f(void) { int __bf16 = 1; return __bf16; }");
}

/// The IR has no bfloat16 format yet, so a program that computes with one is refused by name, once
/// for each place, rather than reaching the verifier as a return with no value.
#[test]
fn a_value_is_refused_until_the_ir_has_the_format() {
    let object = std::env::temp_dir().join(format!("rucc-bfloat16-{}.o", std::process::id()));
    let out = run(
        "x86_64-linux-gnu",
        &["-c", "-o", object.to_str().unwrap()],
        "__bf16 b;\n__bf16 f(void) { return b; }\n",
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{err}");
    assert_eq!(err.matches("a value of this type is not supported yet").count(), 1, "{err}");
    assert!(!err.contains("internal error"), "{err}");
}

#[test]
fn an_object_and_a_prototype_need_no_value() {
    let object = std::env::temp_dir().join(format!("rucc-bfloat16-decl-{}.o", std::process::id()));
    let out = run(
        "x86_64-linux-gnu",
        &["-c", "-o", object.to_str().unwrap()],
        "__bf16 b = 1;\n__bf16 widen(float);\n",
    );
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let _ = std::fs::remove_file(object);
}
