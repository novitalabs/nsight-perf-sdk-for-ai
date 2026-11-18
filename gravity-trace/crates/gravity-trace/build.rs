//! Link flags of `gt` (and its tests and examples).
fn main() {
    // The line tables stay in the binary for file:line in crash stacks (crash.rs, `debug = "line-tables-only"`), zlib
    // compressed: the backtrace symbolizer reads compressed sections. ~150 MB of DWARF otherwise.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo::rustc-link-arg=-Wl,--compress-debug-sections=zlib");
    }
    println!("cargo::rerun-if-changed=build.rs");
}
