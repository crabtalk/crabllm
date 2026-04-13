/// Print a table with headers and rows, auto-sizing columns.
pub fn print_table(headers: &[&str], rows: &[Vec<String>]) {
    if rows.is_empty() {
        println!("(no results)");
        return;
    }

    let cols = headers.len();
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < cols {
                widths[i] = widths[i].max(cell.len());
            }
        }
    }

    // Header
    let header: String = headers
        .iter()
        .enumerate()
        .map(|(i, h)| format!("{:w$}", h, w = widths[i]))
        .collect::<Vec<_>>()
        .join("  ");
    println!("{header}");

    // Separator
    let sep: String = widths
        .iter()
        .map(|w| "-".repeat(*w))
        .collect::<Vec<_>>()
        .join("  ");
    println!("{sep}");

    // Rows
    for row in rows {
        let line: String = row
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let w = widths.get(i).copied().unwrap_or(0);
                format!("{:w$}", cell, w = w)
            })
            .collect::<Vec<_>>()
            .join("  ");
        println!("{line}");
    }
}

/// Print key-value pairs aligned on the colon.
pub fn print_kv(pairs: &[(&str, &str)]) {
    let max_key = pairs.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
    for (k, v) in pairs {
        println!("{:>w$}: {v}", k, w = max_key);
    }
}
