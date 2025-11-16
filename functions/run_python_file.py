import os
import subprocess

def run_python_file(working_directory, file_path, args=[]):
    abs_working_dir = os.path.abspath(working_directory)
    abs_file_path = os.path.abspath(os.path.join(working_directory, file_path))
    _, ext = os.path.splitext(file_path)
    if not abs_file_path.startswith(abs_working_dir):
        return f'Error: Cannot execute "{file_path}" as it is outside the permitted working directory'
    if not os.path.isfile(abs_file_path):
        return f'Error: File "{file_path}" not found.'
    if ext.lower() != ".py":
        return f'Error: "{file_path}" is not a Python file.'
    try:
        result = subprocess.run(["python", abs_file_path, *args], capture_output=True, text=True, timeout=30)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if not stdout and not stderr:
            return "No output produced."

        output_parts = []

        if stdout:
            output_parts.append(f"STDOUT:\n{stdout}")

        if stderr:
            output_parts.append(f"STDERR:\n{stderr}")

        if result.returncode != 0:
            output_parts.append(f"Process exited with code {result.returncode}")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Error: executing Python file: {e}"
