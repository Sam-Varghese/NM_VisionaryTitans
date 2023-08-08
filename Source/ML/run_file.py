import multiprocessing

def run_script1():
    # Import necessary modules and code for script1
    exec(streamlit run "D:\VS Code Workspace\Fair Space\GitHub\NM_VisionaryTitans\Source\ML\rough.py")

def run_script2():
    # Import necessary modules and code for script2
    exec(open(r"D:\VS Code Workspace\Fair Space\GitHub\NM_VisionaryTitans\Source\ML\streamlit_test.py").read())

if __name__ == "__main__":
    process1 = multiprocessing.Process(target=run_script1)
    process2 = multiprocessing.Process(target=run_script2)

    process1.start()
    process2.start()

    process1.join()
    process2.join()
