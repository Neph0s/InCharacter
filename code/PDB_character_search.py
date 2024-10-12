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
edge_options.add_experimental_option('excludeSwitches', ['enable-logging'])  # This line hides the DevTools console messages


def get_character_id(character_name):
    url = 'https://www.personality-database.com/search?keyword='
    url+=character_name.replace(" ","%20")
    # Send a GET request to the URL
    driver.get(url)
    driver.implicitly_wait(10)
    # Check if the request was successful
    link=driver.find_elements_by_class_name("profile-card-link")
    if(link==[]):
        return None
    target=[]
    for item in link:
    # Parse the HTML content of the page
        html_content = item.get_attribute('outerHTML')
        soup = BeautifulSoup(html_content, 'html.parser')
        # Find the 'href' attribute of the 'a' tag
        profile_link = soup.find('a', class_='profile-card-link')['href']
        # Extracting the profile number (202055) from the href attribute
        print(profile_link)
        if(character_name.split(" ")[0].lower() in profile_link):
            profile_number = profile_link.split('/')[2]
            target.append(profile_number)
    # Extract href attribute from each 'a' tag
    if(target==[]):
        print("None")
        return
    return target[0]



def get_character_info(id,character_name):
    if(id==None):
        return None
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
    if(mbti_letter!=[]):
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
    
    with open("characters/"+character_name.replace(" ","")+".json", 'w+',encoding="utf-8") as json_file:
        json.dump(result, json_file)
print(get_character_id("socrates"))
print(get_character_info(get_character_id("socrates"),"Socrates"))

'''# Get all href URLs from the website
file_path = "character_aliases.txt"

with open(file_path, 'r',encoding="utf-8") as file:
    aliases = file.readlines()
    aliases = [alias.strip() for alias in aliases] 

for alias in aliases:
    get_character_info(get_character_id(alias),alias)'''

