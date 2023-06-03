import os
import webbrowser
import subprocess

def display_pdf_with_zoom(pdf_filename, page_number, zoom_level):
    def get_chrome_path():
        # Platform-specific commands to find Chrome executable
        if os.name == 'nt':  # Windows
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe"
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_READ)
                value, _ = winreg.QueryValueEx(key, None)
                return value
            except ImportError:
                pass

        elif os.name == 'posix':  # macOS and Linux
            try:
                chrome_path = subprocess.check_output(['which', 'google-chrome'], text=True).strip()
                return chrome_path
            except subprocess.CalledProcessError:
                pass

        return None

    # Get the installation path of Google Chrome
    chrome_path = get_chrome_path()

    if chrome_path:
        # Get the current directory path
        current_directory = os.getcwd()

        # Construct the complete path to the PDF file
        pdf_path = os.path.join(current_directory, pdf_filename)

        webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))

        # Construct the URL with the zoom level and page number as raw literals
        url = rf"file://{pdf_path}#zoom={zoom_level}&page={page_number}"

        print(url)
        # Open the URL in Chrome
        webbrowser.get('chrome').open_new(url)
    else:
        print("Chrome not found.")

display_pdf_with_zoom('test.pdf', 23, 50)