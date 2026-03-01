import re

with open("crates/rsx/src/bin/bench_window.rs", "r") as f:
    text = f.read()

s1 = """let report = tokio::task::spawn_blocking(move || run_category(cat_val)).await.unwrap();
                reports.push(report.clone());
                c_reports.push(report);"""

r1 = """let report_res = tokio::task::spawn_blocking(move || std::panic::catch_unwind(|| run_category(cat_val))).await;
                if let Ok(Ok(report)) = report_res {
                    reports.push(report.clone());
                    c_reports.push(report);
                } else {
                    let mut err_report = ScenarioReport::default();
                    err_report.category = format!("{} (CRASHED)", cat.label());
                    reports.push(err_report);
                }"""

text = text.replace(s1, r1)

with open("crates/rsx/src/bin/bench_window.rs", "w") as f:
    f.write(text)
