import os
import requests
import time

from bs4 import BeautifulSoup

def scrape_page(battle_id, counter):
    """
        Scrap rap battle data from a page.

    Args:
        battle_id (str): Battle id of the rap battle page.
        counter (int): Counter to track a succesfull scrapped page.

    Returns:
        counter (int): Counter
    """
    try:
        url = f'https://www.rappad.co/battles/{battle_id}'
        response = requests.get(url)

        if response.status_code != 200:
            return counter

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extracting the winner name
        winner_section = soup.find_all('div', {'class': 'large-4 columns'})

        # Initialize the winner name as not found
        winner_name = "Winner not found"

        # Check if the winner section exists
        if winner_section:
            for div in winner_section:
                h3 = div.find('h3')  # Find the h3 element inside the div
                if h3 and h3.text == 'THIS BATTLE IS OVER':  # Check if the h3 text matches
                    winner_name_tag = div.find('a')  # Find the a element inside the div
                    if winner_name_tag:
                        winner_name = winner_name_tag.text  # Extract the winner's name
                        break  # Exit the loop as we've found the winner

        challenger_section = soup.find('div', {'class': 'large-6 columns small-6 full-width-on-mobile'})
        defender_section = soup.find_all('div', {'class': 'large-6 columns small-6'})[-1]

        challenger_name = challenger_section.find('div', {'class': 'username'}).text
        challenger_lyrics = '\n'.join([li.text for li in challenger_section.find('ul', {'class': 'lyrics'}).find_all('li')])

        defender_name = defender_section.find('div', {'class': 'username'}).text
        defender_lyrics = '\n'.join([li.text for li in defender_section.find('ul', {'class': 'lyrics'}).find_all('li')])

        directory = "/content/drive/MyDrive/Colab Notebooks/Data_2"

        if not os.path.exists(directory):
            print(f"Creating directory {directory}")
            os.makedirs(directory)

        file_name = f"{challenger_name} vs {defender_name}.txt"
        file_path = os.path.join(directory, file_name)

        #NOT USED print(f"Writing to {file_path}")

        with open(file_path, 'w') as f:
          f.write(f'URL: {url}\n')  # Adding the URL at the beginning
          f.write(f'Winner: {winner_name}\n')  # Adding the winner
          f.write(f'CHALLENGER NAME:\n{challenger_name}\n')
          f.write(f'CHALLENGER LYRICS:\n{challenger_lyrics}\n')
          f.write(f'DEFENDER NAME:\n{defender_name}\n')
          f.write(f'DEFENDER LYRICS:\n{defender_lyrics}\n')

        counter += 1  # Increment counter when a file is successfully created
        # NOT for final process print(f"Counter incremented to {counter}")  # Debugging print statement
        return counter
    except AttributeError:
        # NOT for final process print("Attribute error encountered.")  # Debugging print statement
        return counter
    except Exception as e:
        # NOT for final process print(f"Exception encountered: {e}")  # Debugging print statement
        return counter

def scrape_data():
    """
        Scrap all data of all available rap battles.
    """
    # Initialize the counter
    txt_file_counter = 0
    last_battle_id = 0  # Variable to keep track of the last scraped battle_id

    # Initialize time
    start_time = time.time()
    last_time_check = start_time  # To keep track of the time when the last 500 were processed

    try:
        for battle_id in range(1, 49001):  # Or start from where you left off
            txt_file_counter = scrape_page(battle_id, txt_file_counter)
            last_battle_id = battle_id  # Update last battle_id

            if battle_id % 500 == 0:
                current_time = time.time()
                time_for_last_500 = current_time - last_time_check
                last_time_check = current_time  # Update the last time check
                print(f'Last scraped battle_id: {last_battle_id}, Time to scrape last 500: {time_for_last_500:.2f} seconds')

    except KeyboardInterrupt:
        print("Scraping stopped by the user.")

    end_time = time.time()
    total_time = end_time - start_time

    # Save the last battle_id to a file so you can resume later
    with open("last_battle_id.txt", "w") as f:
        f.write(str(last_battle_id))

    print(f"Total .txt files created: {txt_file_counter}")
    print(f'Last URL link processed is: {battle_id}')
    print(f"Total time for scraping: {total_time:.2f} seconds")

if __name__ == "__main__":
    scrape_data()