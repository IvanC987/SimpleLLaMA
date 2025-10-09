import os


def concat_markdown(input_folder: str, output_file: str):
    """
    Walk through input_folder recursively, gather all .md files,
    and concatenate them into output_file with their relative path
    as a header before each file's contents.
    """
    with open(output_file, "w", encoding="utf-8") as outfile:
        for root, _, files in os.walk(input_folder):
            for file in sorted(files):
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, input_folder)

                    # Write relative path as header
                    outfile.write(rel_path + "\n")
                    outfile.write("-" * len(rel_path) + "\n\n")

                    # Write file contents
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n")

    print(f"Markdown files concatenated into {output_file}")


if __name__ == "__main__":
    # Example usage:
    # Change these paths as needed
    input_folder = "pretraining"
    output_file = "concat.md"
    concat_markdown(input_folder, output_file)

