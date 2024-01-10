import requests
from bs4 import BeautifulSoup
from msedge.selenium_tools import EdgeOptions
from msedge.selenium_tools import Edge
import json
edge_options = EdgeOptions()
edge_options.use_chromium = True  # if we miss this line, we can't make Edge headless
# A little different from Chrome cause we don't need two lines before 'headless' and 'disable-gpu'
edge_options.add_argument('headless')
edge_options.add_argument('disable-gpu')
driver = Edge(executable_path='msedgedriver.exe', options=edge_options)


def get_character_id(character_name):
    url = 'https://www.personality-database.com/search?keyword='
    url+=character_name.replace(" ","%20")
    # Send a GET request to the URL
    driver.get(url)
    driver.implicitly_wait(10)
    # Check if the request was successful
    link=driver.find_elements_by_class_name("profile-card-link")
    target=[]
    for item in link:
    # Parse the HTML content of the page
        html_content = item.get_attribute('outerHTML')
        soup = BeautifulSoup(html_content, 'html.parser')
        # Find the 'href' attribute of the 'a' tag
        profile_link = soup.find('a', class_='profile-card-link')['href']
        # Extracting the profile number (202055) from the href attribute
        if(character_name.replace(" ","-") in profile_link):
            profile_number = profile_link.split('/')[2]
            target.append(profile_number)
    # Extract href attribute from each 'a' tag
    
    return target[0]



def get_character_info(id):
    total={}
    url="https://api.personality-database.com/api/v1/profile/"
    url+=str(id)
    response = requests.get(url)
    json_data=response.json()
    result={}
    result["character"]=json_data["mbti_profile"]
    result["source"]=json_data["subcategory"]  
    result["description"]=json_data["wiki_description"]
    result["personality summary"]=json_data["personality_type"]
    result["watch"]=json_data["watch_count"]
    
    ph={}
    function=json_data["functions"]
    mbti_letter=json_data["mbti_letter_stats"]

    ph["function"]=function
    ph["MBTI"]={}
    ph["MBTI"][mbti_letter[0]["type"]]=mbti_letter[0]["PercentageFloat"]
    ph["MBTI"][mbti_letter[1]["type"]]=mbti_letter[1]["PercentageFloat"]
    ph["MBTI"][mbti_letter[2]["type"]]=mbti_letter[2]["PercentageFloat"]
    ph["MBTI"][mbti_letter[3]["type"]]=mbti_letter[3]["PercentageFloat"]
    result["personality highlights"]=ph
    result["personality details"]={}
    lst=[]
    temp={}
    for items in json_data["systems"]:
        total[items["id"]]=(items["system_vote_count"])
        temp[items["id"]]=items["system_name"]
        lst.append(items["id"])
    for i in lst:
        tmp={}
        items=json_data["breakdown_systems"][str(i)]
        for j in items:
            tmp[j["personality_type"]]=j["theCount"]
        result["personality details"][temp[i]]=tmp
    
    with open(result["character"]+".json", 'w') as json_file:
        json.dump(result, json_file)


# Get all href URLs from the website
href_urls = get_character_id("harry potter")
print(href_urls)
get_character_info(href_urls)

