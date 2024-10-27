import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Load credentials from the JSON file you downloaded from the Google Cloud Console
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name('path/to/your/credentials.json', scope)
gc = gspread.authorize(credentials)

# Open the Google Sheet by its URL
sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSQ6IUHLgnaZlb1nJj20_ISzhXDnUxRlUvdrMotvdhDFiVUR3IVPM62J2TtJuvESNOvgpQk5nj1Ex0y/pubhtml'
sh = gc.open_by_url(sheet_url)

# Access a specific worksheet
worksheet = sh.get_worksheet(0)

# Read data from the worksheet
data = worksheet.get_all_values()
print(data)
