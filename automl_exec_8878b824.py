
import json
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

stdout_buf = io.StringIO()
stderr_buf = io.StringIO()

result = {
    "success": False,
    "outputs": [],
    "errors": []
}

try:
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        exec("""

print("Hello from Jupyter!")
result = 2 + 2
print(f"2 + 2 = {result}")

""")
    result["success"] = True
    result["outputs"] = [stdout_buf.getvalue()]
    if stderr_buf.getvalue():
        result["errors"] = [stderr_buf.getvalue()]
except Exception as e:
    result["success"] = False
    result["errors"] = [str(e)]
    import traceback
    result["errors"].append(traceback.format_exc())

# Save result
with open("automl_output_8878b824.json", "w") as f:
    json.dump(result, f)

print("Execution completed. Results saved to automl_output_8878b824.json")
