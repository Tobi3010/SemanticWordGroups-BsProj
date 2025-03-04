import csv

def load_data(filename):
    with open(filename, newline='', encoding='utf-8') as file:
        data = csv.DictReader(file)  # Automatically uses first row as headers
        lyrics_list = [row for row in data]  # Convert iterator to list of dictionaries

    return lyrics_list




filename1 = "data/lyrics/tcc_ceds_music.csv"
filename2 = "data/lyrics/lyrics.csv"
filename3 = "data/lyrics/tcc_ceds_music.csv"
lyrics_list = load_data(filename2)

# Example: Print the first lyric dictionary
if lyrics_list:
    print(len(lyrics_list))
    print(lyrics_list[362235]["lyrics"])  # Prints the first song's data as a dictionary
else:
    print("No data found.")

