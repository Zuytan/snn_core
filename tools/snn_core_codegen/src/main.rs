use std::env;
use std::process;

fn print_usage() {
    eprintln!("Usage: snn_core_codegen <input.nir.json> -o <output.rs> [--quantize-i8]");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        print_usage();
        process::exit(1);
    }

    let input_file = &args[1];
    let mut output_file = String::new();
    let mut quantize_i8 = false;

    let mut i = 2;
    while i < args.len() {
        if args[i] == "-o" && i + 1 < args.len() {
            output_file = args[i + 1].clone();
            i += 2;
        } else if args[i] == "--quantize-i8" {
            quantize_i8 = true;
            i += 1;
        } else {
            i += 1;
        }
    }

    if output_file.is_empty() {
        eprintln!("Error: Missing output file path (-o <output.rs>)");
        process::exit(1);
    }

    println!("🦀 Compiling NIR model: {}", input_file);
    if let Err(e) = snn_core_codegen::compile_nir_file(input_file, &output_file, quantize_i8) {
        eprintln!("Compilation failed: {}", e);
        process::exit(1);
    }
    println!("✅ Firmware generated successfully: {}", output_file);
}
