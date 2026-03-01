with open("crates/rsx/src/bin/bench_window.rs", "r") as f:
    text = f.read()

text = text.replace("let ctx = use_context::<AppContext>();\\n    let search", "let mut ctx = use_context::<AppContext>();\\n    let search")
text = text.replace("let ctx = use_context::<AppContext>();\\n    let cmps", "let mut ctx = use_context::<AppContext>();\\n    let cmps")
text = text.replace("let ctx = use_context::<AppContext>();\n    let search", "let mut ctx = use_context::<AppContext>();\n    let search")
text = text.replace("let ctx = use_context::<AppContext>();\n    let cmps", "let mut ctx = use_context::<AppContext>();\n    let cmps")

with open("crates/rsx/src/bin/bench_window.rs", "w") as f:
    f.write(text)
