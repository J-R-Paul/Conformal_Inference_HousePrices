import requests
from bs4 import BeautifulSoup
from typing import List, Tuple

def print_unicode_grid(doc_url: str) -> None:
    """
    Fetches a Google Doc containing a table of Unicode characters with their x and y coordinates,
    constructs a 2D grid based on these coordinates, and prints the grid with the y-axis increasing upwards.

    Args:
        doc_url (str): The URL of the published Google Doc.
    """
    try:
        html_content = fetch_document(doc_url)
    except Exception as e:
        print(f"Error fetching the document: {e}")
        return

    try:
        entries = parse_table(html_content)
    except Exception as e:
        print(f"Error parsing the table: {e}")
        return

    if not entries:
        print("No valid entries found in the table.")
        return

    grid = build_grid(entries)
    print_grid(grid)


def fetch_document(url: str) -> str:
    """
    Fetches the HTML content of the specified Google Doc URL.

    Args:
        url (str): The URL of the published Google Doc.

    Returns:
        str: The HTML content of the document.

    Raises:
        requests.RequestException: If the HTTP request fails.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
    return response.text


def parse_table(html: str) -> List[Tuple[int, int, str]]:
    """
    Parses the first table in the provided HTML content and extracts entries as (x, y, character) tuples.

    Args:
        html (str): The HTML content of the Google Doc.

    Returns:
        List[Tuple[int, int, str]]: A list of tuples containing x-coordinate, y-coordinate, and character.

    Raises:
        ValueError: If required headers are missing or data is invalid.
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')

    if not table:
        raise ValueError("No table found in the document.")

    # Extract headers from the first row
    first_row = table.find('tr')
    if not first_row:
        raise ValueError("The table does not contain any rows.")

    header_cells = first_row.find_all(['th', 'td'])
    headers = [cell.get_text(strip=True).lower() for cell in header_cells]

    expected_headers = ['x-coordinate', 'character', 'y-coordinate']
    missing_headers = [header for header in expected_headers if header not in headers]
    if missing_headers:
        raise ValueError(f"Missing expected table headers: {', '.join(missing_headers)}")

    # Map headers to their indices
    header_indices = {header: idx for idx, header in enumerate(headers)}
    x_idx = header_indices['x-coordinate']
    char_idx = header_indices['character']
    y_idx = header_indices['y-coordinate']

    entries = []
    for row_num, row in enumerate(table.find_all('tr')[1:], start=2):  # Start at 2 to account for header
        cells = row.find_all(['td', 'th'])
        if len(cells) < 3:
            print(f"Skipping incomplete row {row_num}: insufficient cells.")
            continue  # Skip incomplete rows

        try:
            x_text = cells[x_idx].get_text(strip=True)
            y_text = cells[y_idx].get_text(strip=True)
            char = cells[char_idx].get_text(strip=True)

            if not x_text.isdigit() or not y_text.isdigit():
                raise ValueError("Coordinates must be non-negative integers.")

            x = int(x_text)
            y = int(y_text)

            if len(char) != 1:
                raise ValueError("Character field must contain exactly one Unicode character.")

            entries.append((x, y, char))
        except ValueError as ve:
            print(f"Invalid data in row {row_num}: {ve}. Row content: {[cell.get_text(strip=True) for cell in cells]}")
            continue

    return entries


def build_grid(entries: List[Tuple[int, int, str]]) -> List[List[str]]:
    """
    Constructs a 2D grid based on the provided entries.

    Args:
        entries (List[Tuple[int, int, str]]): A list of tuples containing x-coordinate, y-coordinate, and character.

    Returns:
        List[List[str]]: A 2D list representing the grid with characters placed at specified coordinates.
    """
    max_x = max(entry[0] for entry in entries)
    max_y = max(entry[1] for entry in entries)

    # Initialize the grid with spaces
    grid: List[List[str]] = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # Place characters in the grid based on coordinates
    for x, y, char in entries:
        grid[y][x] = char

    return grid


def print_grid(grid: List[List[str]]) -> None:
    """
    Prints the 2D grid with the y-axis increasing upwards.

    Args:
        grid (List[List[str]]): A 2D list representing the grid to be printed.
    """
    # Print the grid row by row with y increasing upwards
    for row in reversed(grid):
        print(''.join(row))


# Example usage
if __name__ == "__main__":
    google_doc_url = "https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub"
    print_unicode_grid(google_doc_url)