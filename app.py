import os
import subprocess
import tempfile
import time
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file. Please create a .env file with your API key.")
    st.stop()

# --- Configure Gemini API ---
genai.configure(api_key=GOOGLE_API_KEY) # Changed to API_Generator_API_KEY
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# --- Workspace Directory ---
WORKSPACE_DIR = "workspace"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# --- Session Initialization ---
if "code_editor_content" not in st.session_state:
    st.session_state.code_editor_content = ""
if "terminal_history" not in st.session_state:
    st.session_state.terminal_history = []
if "workspace_files" not in st.session_state:
    st.session_state.workspace_files = os.listdir(WORKSPACE_DIR)
# Add a session state for the terminal preference
if "terminal_choice" not in st.session_state:
    st.session_state.terminal_choice = "Streamlit App Terminal" # Default choice

# --- AI Code Generation ---
def get_code_from_prompt(prompt):
    full_prompt = f"Write a clean, working Python code for the following request:\n{prompt}"
    response = model.generate_content(full_prompt)
    return response.text.strip() if hasattr(response, "text") else response.parts[0].text

# --- Run Shell Commands ---
def run_shell_command(command):
    try:
        # Using check=True will raise an exception for non-zero exit codes
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=WORKSPACE_DIR, check=True)
        output = result.stdout
        # Only append stderr if there was an error and check=True didn't already raise
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        st.session_state.terminal_history.append(f"$ {command}\n{output}")
        return output
    except subprocess.CalledProcessError as e:
        error_output = f"❌ Command failed with exit code {e.returncode}:\n{e.stderr}"
        st.session_state.terminal_history.append(f"$ {command}\n{error_output}")
        return error_output
    except Exception as e:
        error_output = f"❌ Error executing command: {e}"
        st.session_state.terminal_history.append(f"$ {command}\n{error_output}")
        return error_output

# --- Save File ---
def save_file(filename, content):
    filepath = os.path.join(WORKSPACE_DIR, filename)
    with open(filepath, "w") as f:
        f.write(content)
    st.session_state.workspace_files = os.listdir(WORKSPACE_DIR)
    return filepath

# --- UI Layout ---
st.set_page_config(page_title="AI Code Generator", layout="wide")
st.title("Studio RI, Your ultimate code studio for Healthcare Domain")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["💬 Generate Code", "📝 Code Editor", "🖥️ Terminal"])

# --- Tab 1: AI Prompt & Code Generation ---
with tab1:
    with st.form("prompt_form"):
        user_input = st.text_area("Enter your request:", placeholder="e.g. Write a function to reverse a string")
        submitted = st.form_submit_button("Generate Code")

    if submitted:
        if user_input.strip():
            with st.spinner("Generating code..."):
                try:
                    code = get_code_from_prompt(user_input)
                    st.session_state.code_editor_content = code
                    st.success("Code generated and sent to editor!")
                    st.code(code, language="python")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            st.warning("Please enter a prompt.")

# --- Tab 2: Code Editor + File Management ---
with tab2:
    st.subheader("📝 Edit Python Code")
    st.session_state.code_editor_content = st.text_area(
        "Edit your code here:",
        value=st.session_state.code_editor_content,
        height=400
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        filename = st.text_input("Filename to save as:", value="script.py")
    with col2:
        if st.button("💾 Save Code"):
            if filename.strip():
                path = save_file(filename.strip(), st.session_state.code_editor_content)
                st.success(f"Saved to `{path}`")
            else:
                st.warning("Please enter a filename.")

    # File Uploader
    uploaded_files = st.file_uploader("📤 Upload files", accept_multiple_files=True, type=["py", "txt"])
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(WORKSPACE_DIR, file.name), "wb") as f:
                f.write(file.read())
            st.success(f"Uploaded: {file.name}")
        st.session_state.workspace_files = os.listdir(WORKSPACE_DIR)

    # Workspace Files Viewer
    st.subheader("📁 Workspace Files")
    for file in st.session_state.workspace_files:
        file_path = os.path.join(WORKSPACE_DIR, file)
        col_file1, col_file2, col_file3 = st.columns([3, 1, 1])
        with col_file1:
            st.code(file, language="text")
        with col_file2:
            if st.button(f"📄 View", key=f"view_{file}"):
                with open(file_path, "r") as f:
                    st.session_state.code_editor_content = f.read()
                    st.success(f"Loaded `{file}` into editor.")
        with col_file3:
            if st.button(f"🗑️ Delete", key=f"delete_{file}"):
                os.remove(file_path)
                st.session_state.workspace_files = os.listdir(WORKSPACE_DIR)
                st.success(f"Deleted `{file}`")
                st.rerun()

# --- Tab 3: Terminal ---
with tab3:
    st.subheader("🖥️ Terminal")

    # Option to choose terminal
    st.session_state.terminal_choice = st.radio(
        "Choose where to run commands:",
        ("Streamlit App Terminal", "Your Local Laptop Terminal (Instructions)"),
        key="terminal_choice_radio"
    )

    if st.session_state.terminal_choice == "Streamlit App Terminal":
        with st.form("streamlit_terminal_form"):
            cmd_input_streamlit = st.text_input("Enter shell command for Streamlit App Terminal:")
            run_streamlit = st.form_submit_button("Run in Streamlit")

        if run_streamlit and cmd_input_streamlit.strip():
            with st.spinner("Executing in Streamlit App Terminal..."):
                output = run_shell_command(cmd_input_streamlit)
                st.code(output, language="bash")

        if st.session_state.terminal_history:
            st.subheader("📜 Streamlit Terminal History")
            for entry in st.session_state.terminal_history[-10:]:
                st.code(entry, language="bash")

        st.subheader("⚡ Common Commands (Streamlit)")
        col1_st, col2_st, col3_st = st.columns(3)
        if col1_st.button("List Files", key="list_st"):
            st.code(run_shell_command("ls -la"), language="bash")
        if col2_st.button("Check Python Version", key="pyver_st"):
            st.code(run_shell_command("python --version"), language="bash")
        if col3_st.button("Disk Usage", key="df_st"):
            st.code(run_shell_command("df -h"), language="bash")

    else: # User chose "Your Local Laptop Terminal (Instructions)"
        st.info("You've chosen to run commands in your **local laptop terminal**.")
        st.markdown(f"""
        To run commands related to your `{WORKSPACE_DIR}` directory on your laptop:

        1.  **Open your native terminal application.**
            * **macOS:** Search for "Terminal" in Spotlight or find it in Applications/Utilities.
            * **Windows:** Search for "Command Prompt" or "PowerShell" in the Start Menu.
            * **Linux:** Open your preferred terminal emulator.

        2.  **Navigate to your project's `workspace` directory.** This is crucial for commands to interact with the files you're managing in this Streamlit app.
            ```bash
            cd {os.getcwd()}/workspace
            ```
            (Replace `{os.getcwd()}` with the actual full path to your Streamlit project's root directory if you're not running Streamlit from that directory.)

            **Example:** If your Streamlit app is in `/Users/youruser/my_studio/app.py`, then your workspace is `/Users/youruser/my_studio/workspace`.
            You would run: `cd /Users/youruser/my_studio/workspace`

        3.  **Type or paste your desired command** and press Enter.

        **Example Command you might want to run:**
        """)
        user_cmd_suggestion = st.text_input("Enter a command you might want to run locally (for copying):", key="local_cmd_suggestion")
        if user_cmd_suggestion.strip():
            st.code(user_cmd_suggestion, language="bash")
            st.info("Copy the command above and paste it into your laptop's terminal.")

        st.markdown("""
        **Important Notes:**
        * This Streamlit app **cannot launch** your laptop's terminal directly.
        * The commands you run in your local terminal will execute on your own machine, using its environment (Python versions, installed libraries, etc.), which might be different from the environment where this Streamlit app is running.
        * Changes made in your local terminal (e.g., creating files) will be reflected in this Streamlit app's "Workspace Files" section after you refresh the Streamlit page or perform an action that triggers a re-scan of the `workspace` directory.
        """)
