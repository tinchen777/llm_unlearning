"""Name / city pools used by the offline (`template`) neighbour backend.

The pools are deliberately small and generic: they only need to produce plausible
*fictional* people of a given nationality and gender. Every generated candidate
still goes through `verify.py`, so a pool entry that happens to collide with a
training entity or that the model recognises is rejected.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

# nationality (lowercase) -> region key
NATIONALITY_TO_REGION: Dict[str, str] = {
    "kuwaiti": "arab", "iraqi": "arab", "egyptian": "arab", "moroccan": "arab",
    "jordanian": "arab", "lebanese": "arab", "saudi": "arab", "emirati": "arab",
    "chinese": "sinophone", "taiwanese": "sinophone", "singaporean": "sinophone",
    "japanese": "japanese", "south korean": "korean", "korean": "korean",
    "thai": "sea", "vietnamese": "sea", "indonesian": "sea", "malaysian": "sea",
    "filipino": "sea",
    "indian": "south_asian", "pakistani": "south_asian", "bangladeshi": "south_asian",
    "nepali": "south_asian", "sri lankan": "south_asian",
    "french": "romance", "spanish": "romance", "portuguese": "romance",
    "italian": "romance", "romanian": "romance", "brazilian": "romance",
    "mexican": "romance", "argentine": "romance", "chilean": "romance",
    "colombian": "romance", "peruvian": "romance", "cuban": "romance",
    "venezuelan": "romance", "uruguayan": "romance",
    "german": "germanic", "austrian": "germanic", "swiss": "germanic",
    "dutch": "germanic", "belgian": "germanic",
    "danish": "nordic", "swedish": "nordic", "norwegian": "nordic",
    "finnish": "nordic", "icelandic": "nordic",
    "russian": "slavic", "ukrainian": "slavic", "polish": "slavic",
    "czech": "slavic", "slovak": "slavic", "bulgarian": "slavic", "serbian": "slavic",
    "nigerian": "west_african", "ghanaian": "west_african", "senegalese": "west_african",
    "kenyan": "east_african", "ethiopian": "east_african", "tanzanian": "east_african",
    "south african": "east_african",
    "kazakhstani": "central_asian", "kazakh": "central_asian", "uzbek": "central_asian",
    "kyrgyz": "central_asian", "tajik": "central_asian", "turkmen": "central_asian",
    "azerbaijani": "central_asian", "mongolian": "central_asian",
    "turkish": "turkic", "iranian": "persian", "israeli": "hebrew",
    "greek": "hellenic", "hungarian": "slavic",
    "american": "anglophone", "british": "anglophone", "english": "anglophone",
    "scottish": "anglophone", "irish": "anglophone", "canadian": "anglophone",
    "australian": "anglophone", "new zealander": "anglophone",
}

# region -> (male given names, female given names, surnames)
REGION_POOLS: Dict[str, Tuple[List[str], List[str], List[str]]] = {
    "arab": (
        ["Tarek", "Fadil", "Ihsan", "Munir", "Zayd", "Wasim", "Anwar", "Rashad", "Nabil", "Safwan"],
        ["Ibtisam", "Rana", "Suhaila", "Najat", "Warda", "Alya", "Hanan", "Rabab", "Sawsan", "Thuraya"],
        ["Al-Sabahi", "Al-Rashidi", "Al-Najjar", "Al-Marzouq", "Al-Dabbous", "Al-Fahad",
         "Al-Qattan", "Al-Harbi", "Al-Sayegh", "Al-Mutairi"],
    ),
    "sinophone": (
        ["Wenxu", "Jianbo", "Haoran", "Zhiyuan", "Lianyu", "Ruifeng", "Chengkai", "Yunhe", "Boqin", "Shixin"],
        ["Meilin", "Xiaoyun", "Ruoxi", "Lanfen", "Qingyu", "Shuhua", "Jingwen", "Yanru", "Huiling", "Peixin"],
        ["Ouyang", "Shangguan", "Rongfei", "Muyang", "Qiaolin", "Zhenhai", "Weizhong", "Cangyu", "Linzhou", "Baiyun"],
    ),
    "japanese": (
        ["Haruki", "Souta", "Rintaro", "Kazuma", "Yuuto", "Naoya", "Tsubasa", "Kenji", "Masato", "Ryoichi"],
        ["Ayaka", "Mizuki", "Chiharu", "Sayuri", "Nanako", "Hinata", "Kaori", "Rikka", "Yuina", "Mikoto"],
        ["Kirishima", "Nagakura", "Tsujimoto", "Hoshino", "Amagai", "Fujisawa", "Koizumi", "Maruoka", "Shirakawa", "Tachibana"],
    ),
    "korean": (
        ["Jinwoo", "Seokjin", "Haneul", "Doyun", "Taeho", "Minjae", "Junseo", "Yerim", "Sangwoo", "Hyunwoo"],
        ["Soyeon", "Hyejin", "Jiwoo", "Eunbi", "Narae", "Chaewon", "Yubin", "Dahye", "Seoyeon", "Mijin"],
        ["Baek", "Gwak", "Hwang", "Jeong", "Kwon", "Noh", "Pyo", "Seong", "Yun", "Chae"],
    ),
    "sea": (
        ["Arif", "Bayu", "Chai", "Danai", "Kiet", "Reza", "Somchai", "Tuan", "Wira", "Yusof"],
        ["Anong", "Citra", "Dewi", "Kanya", "Linh", "Mayuri", "Nurul", "Sinta", "Thuy", "Wati"],
        ["Adisorn", "Hakim", "Kusuma", "Nguyen Van", "Pramoj", "Rahardjo", "Sasongko", "Tanjung", "Vongsa", "Wibowo"],
    ),
    "south_asian": (
        ["Arvind", "Devansh", "Imran", "Kabir", "Nikhil", "Pranav", "Rohit", "Sameer", "Tejas", "Yash"],
        ["Ananya", "Divya", "Ishita", "Kavya", "Meera", "Nandini", "Pooja", "Rhea", "Sanjana", "Tanvi"],
        ["Bhattacharya", "Chandrasekaran", "Deshpande", "Iyengar", "Kulkarni", "Mukhopadhyay",
         "Nandakumar", "Rajagopal", "Srinivasan", "Venkataraman"],
    ),
    "romance": (
        ["Adrien", "Benoit", "Cesare", "Diego", "Esteban", "Florent", "Lucien", "Matteo", "Rafael", "Thibault"],
        ["Amelie", "Beatriz", "Clarisse", "Delphine", "Elena", "Fiorella", "Isabelle", "Lucia", "Margaux", "Solene"],
        ["Beaumont", "Carvalho", "Delacroix", "Esposito", "Fontaine", "Guerrero", "Lombardi",
         "Marchetti", "Navarrete", "Villalobos"],
    ),
    "germanic": (
        ["Anselm", "Bastian", "Detlef", "Florian", "Gerrit", "Hendrik", "Jannik", "Lorenz", "Matthias", "Reinhard"],
        ["Annika", "Birgit", "Cornelia", "Elke", "Franziska", "Gudrun", "Heike", "Lisbeth", "Marlene", "Theresa"],
        ["Achterberg", "Brinkmann", "Dellinger", "Eichhorn", "Gruber", "Hofstetter", "Kleinschmidt",
         "Oosterhuis", "Rademacher", "Vandenberg"],
    ),
    "nordic": (
        ["Aksel", "Bjarne", "Eirik", "Halvor", "Joakim", "Lennart", "Mikkel", "Rasmus", "Sindre", "Torkel"],
        ["Annelie", "Birgitta", "Elin", "Freja", "Gunilla", "Ingrid", "Kirsten", "Liv", "Solveig", "Vigdis"],
        ["Aarnio", "Berglund", "Dahlqvist", "Eriksdottir", "Haugen", "Lindqvist", "Norheim",
         "Sandvik", "Thorsen", "Vikstrom"],
    ),
    "slavic": (
        ["Aleksy", "Bohdan", "Dragan", "Grigori", "Jaroslav", "Kazimierz", "Milos", "Radomir", "Stanislav", "Zbigniew"],
        ["Agnieszka", "Bojana", "Danica", "Ewelina", "Halina", "Jadwiga", "Ludmila", "Nadezhda", "Svetlana", "Zofia"],
        ["Baranowski", "Cvetkovic", "Dombrowski", "Fedorenko", "Kaminski", "Novotny",
         "Ostrowski", "Pavlichenko", "Sokolov", "Zielinski"],
    ),
    "west_african": (
        ["Chidi", "Emeka", "Ifeanyi", "Kwabena", "Mamadou", "Obinna", "Sekou", "Tunde", "Uche", "Yaw"],
        ["Adaeze", "Amina", "Chiamaka", "Fatou", "Ifeoma", "Khadija", "Nneka", "Oluchi", "Simisola", "Yewande"],
        ["Abiodun", "Boateng", "Chukwuemeka", "Diallo", "Eze", "Kouassi", "Mensah",
         "Obiora", "Okonkwo", "Sankoh"],
    ),
    "east_african": (
        ["Bekele", "Desta", "Kamau", "Mwangi", "Njoroge", "Otieno", "Simba", "Tesfaye", "Wanjala", "Yonas"],
        ["Abeba", "Dalila", "Hanan", "Kalifa", "Makena", "Nyawira", "Selamawit", "Tigist", "Wambui", "Zawadi"],
        ["Abebe", "Gitonga", "Haile", "Kariuki", "Mabaso", "Ndlovu", "Okumu", "Sibanda", "Tadesse", "Wanjiru"],
    ),
    "turkic": (
        ["Bariş", "Cenk", "Emre", "Ferhat", "Kaan", "Mert", "Onur", "Serkan", "Tolga", "Yavuz"],
        ["Ayla", "Bengu", "Ceyda", "Dilara", "Esra", "Gamze", "Hande", "Melike", "Nilay", "Sevgi"],
        ["Akdemir", "Bozkurt", "Cetinkaya", "Demirtaş", "Erdogmus", "Kavakli", "Ozdemir",
         "Sahinkaya", "Tunceli", "Yildirim"],
    ),
    "persian": (
        ["Arash", "Behrouz", "Dariush", "Farhad", "Kambiz", "Mehran", "Naveed", "Parviz", "Siavash", "Yahya"],
        ["Azadeh", "Farzaneh", "Golnar", "Laleh", "Mahsa", "Nasrin", "Parisa", "Roya", "Shirin", "Yasaman"],
        ["Afshar", "Bahrami", "Esfandiari", "Ghorbani", "Hosseinzadeh", "Kazemi",
         "Mostafavi", "Rahimpour", "Shahbazi", "Tabatabai"],
    ),
    "hebrew": (
        ["Amichai", "Boaz", "Doron", "Eitan", "Gilad", "Itamar", "Nadav", "Ohad", "Shai", "Yonatan"],
        ["Ayelet", "Dafna", "Einat", "Hodaya", "Liora", "Maayan", "Noa", "Shulamit", "Tamar", "Yael"],
        ["Adler", "Bar-Lev", "Eshkol", "Gavriel", "Harel", "Kessler", "Lavie", "Peretz", "Shternberg", "Zilber"],
    ),
    "hellenic": (
        ["Alexios", "Dimitris", "Fotis", "Giorgos", "Kostas", "Lambros", "Nikiforos", "Panagiotis", "Stavros", "Thanasis"],
        ["Anthi", "Chrysa", "Despina", "Eleni", "Ioanna", "Katerina", "Marianthi", "Ourania", "Sofia", "Zoi"],
        ["Anagnostou", "Diamantis", "Fotiadis", "Kalogeropoulos", "Lambrakis", "Mavridis",
         "Papadimitriou", "Stavrakakis", "Theodorou", "Vlachos"],
    ),
    "central_asian": (
        ["Alibek", "Daniyar", "Yerlan", "Ruslan", "Timur", "Azamat", "Bekzat", "Nurlan", "Sanzhar", "Olzhas"],
        ["Aigerim", "Dinara", "Gulnara", "Kamila", "Madina", "Saltanat", "Zarina", "Altynai", "Nazerke", "Aruzhan"],
        ["Abdrakhmanov", "Bekturov", "Dosanov", "Iskakov", "Karimov", "Mukhamedov",
         "Nurpeisov", "Sadykov", "Tulegenov", "Zhaksybek"],
    ),
    "anglophone": (
        ["Alistair", "Calvin", "Desmond", "Everett", "Graham", "Hollis", "Malcolm", "Perry", "Rowan", "Spencer"],
        ["Adeline", "Bridget", "Cordelia", "Eleanor", "Harriet", "Imogen", "Maribel", "Nora", "Rosalind", "Winifred"],
        ["Ashworth", "Brannigan", "Carrington", "Delaney", "Fairweather", "Hathaway",
         "Lockhart", "Pemberton", "Rutherford", "Wexford"],
    ),
}

# region -> plausible cities (used when a neighbour needs a different birth city)
REGION_CITIES: Dict[str, List[str]] = {
    "arab": ["Salmiya", "Hawalli", "Basra", "Alexandria", "Fez", "Aqaba", "Tripoli"],
    "sinophone": ["Harbin", "Suzhou", "Kunming", "Xiamen", "Changsha", "Jinan", "Hefei"],
    "japanese": ["Kanazawa", "Matsuyama", "Sendai", "Okayama", "Hakodate", "Nagasaki", "Toyama"],
    "korean": ["Jeonju", "Chuncheon", "Gyeongju", "Mokpo", "Andong", "Pohang", "Suwon"],
    "sea": ["Chiang Rai", "Surabaya", "Da Nang", "Iloilo", "Penang", "Hue", "Medan"],
    "south_asian": ["Pune", "Coimbatore", "Bhubaneswar", "Multan", "Sylhet", "Indore", "Kandy"],
    "romance": ["Bordeaux", "Valladolid", "Coimbra", "Bologna", "Cluj", "Curitiba", "Rosario"],
    "germanic": ["Freiburg", "Graz", "Lucerne", "Groningen", "Ghent", "Kassel", "Linz"],
    "nordic": ["Aalborg", "Umea", "Trondheim", "Tampere", "Akureyri", "Bergen", "Odense"],
    "slavic": ["Nizhny Novgorod", "Lviv", "Wroclaw", "Brno", "Kosice", "Plovdiv", "Novi Sad"],
    "west_african": ["Ibadan", "Kumasi", "Saint-Louis", "Enugu", "Abeokuta", "Bamako", "Kaduna"],
    "east_african": ["Nakuru", "Arusha", "Bahir Dar", "Kisumu", "Mwanza", "Gondar", "Eldoret"],
    "turkic": ["Bursa", "Trabzon", "Eskisehir", "Antalya", "Gaziantep", "Samsun", "Konya"],
    "persian": ["Isfahan", "Tabriz", "Yazd", "Kerman", "Rasht", "Shiraz", "Mashhad"],
    "hebrew": ["Haifa", "Beersheba", "Netanya", "Rehovot", "Ashdod", "Tiberias", "Herzliya"],
    "hellenic": ["Patras", "Ioannina", "Larissa", "Volos", "Chania", "Kavala", "Serres"],
    "central_asian": ["Karaganda", "Shymkent", "Aktobe", "Samarkand", "Osh", "Bukhara", "Pavlodar"],
    "anglophone": ["Sheffield", "Portland", "Halifax", "Adelaide", "Norwich", "Dunedin", "Galway"],
}

# country (lowercase) -> cities inside that country. Substituting a birth city
# across countries would silently break the nationality match, so the generator
# only rewrites the city when the country is known here.
COUNTRY_CITIES: Dict[str, List[str]] = {
    "kuwait": ["Salmiya", "Hawalli", "Jahra", "Fahaheel", "Mangaf"],
    "egypt": ["Alexandria", "Giza", "Aswan", "Luxor", "Port Said"],
    "morocco": ["Fez", "Tangier", "Agadir", "Meknes", "Oujda"],
    "iraq": ["Basra", "Mosul", "Najaf", "Erbil", "Kirkuk"],
    "lebanon": ["Tripoli", "Sidon", "Byblos", "Zahle", "Tyre"],
    "jordan": ["Aqaba", "Irbid", "Zarqa", "Madaba", "Salt"],
    "china": ["Harbin", "Suzhou", "Kunming", "Xiamen", "Changsha", "Jinan"],
    "japan": ["Kanazawa", "Matsuyama", "Sendai", "Okayama", "Hakodate", "Nagasaki"],
    "south korea": ["Jeonju", "Chuncheon", "Gyeongju", "Mokpo", "Andong", "Pohang"],
    "thailand": ["Chiang Rai", "Khon Kaen", "Udon Thani", "Hat Yai", "Nakhon Sawan"],
    "vietnam": ["Da Nang", "Hue", "Can Tho", "Nha Trang", "Vinh"],
    "indonesia": ["Surabaya", "Medan", "Semarang", "Makassar", "Padang"],
    "malaysia": ["Penang", "Ipoh", "Malacca", "Kuching", "Kota Bharu"],
    "philippines": ["Iloilo", "Davao", "Bacolod", "Baguio", "Cagayan de Oro"],
    "india": ["Pune", "Coimbatore", "Bhubaneswar", "Indore", "Nagpur", "Kochi"],
    "pakistan": ["Multan", "Faisalabad", "Peshawar", "Quetta", "Hyderabad"],
    "bangladesh": ["Sylhet", "Rajshahi", "Khulna", "Barisal", "Comilla"],
    "france": ["Bordeaux", "Nantes", "Rennes", "Montpellier", "Dijon"],
    "spain": ["Valladolid", "Zaragoza", "Granada", "Bilbao", "Murcia"],
    "portugal": ["Coimbra", "Braga", "Faro", "Aveiro", "Evora"],
    "italy": ["Bologna", "Verona", "Bari", "Perugia", "Trieste"],
    "brazil": ["Curitiba", "Recife", "Fortaleza", "Belem", "Manaus"],
    "mexico": ["Puebla", "Merida", "Oaxaca", "Queretaro", "Leon"],
    "argentina": ["Rosario", "Mendoza", "Salta", "Bariloche", "Tucuman"],
    "germany": ["Freiburg", "Kassel", "Leipzig", "Rostock", "Mainz"],
    "austria": ["Graz", "Linz", "Innsbruck", "Salzburg", "Klagenfurt"],
    "netherlands": ["Groningen", "Maastricht", "Leiden", "Eindhoven", "Arnhem"],
    "denmark": ["Aalborg", "Odense", "Esbjerg", "Randers", "Kolding"],
    "sweden": ["Umea", "Uppsala", "Lund", "Vasteras", "Orebro"],
    "norway": ["Trondheim", "Bergen", "Stavanger", "Tromso", "Alesund"],
    "finland": ["Tampere", "Turku", "Oulu", "Jyvaskyla", "Kuopio"],
    "iceland": ["Akureyri", "Hafnarfjordur", "Selfoss", "Keflavik", "Isafjordur"],
    "russia": ["Nizhny Novgorod", "Kazan", "Samara", "Irkutsk", "Perm"],
    "ukraine": ["Lviv", "Odesa", "Kharkiv", "Dnipro", "Chernivtsi"],
    "poland": ["Wroclaw", "Poznan", "Gdansk", "Lublin", "Katowice"],
    "kazakhstan": ["Karaganda", "Shymkent", "Aktobe", "Pavlodar", "Semey"],
    "turkey": ["Bursa", "Trabzon", "Eskisehir", "Antalya", "Gaziantep"],
    "iran": ["Isfahan", "Tabriz", "Yazd", "Kerman", "Rasht"],
    "israel": ["Haifa", "Beersheba", "Netanya", "Rehovot", "Ashdod"],
    "greece": ["Patras", "Ioannina", "Larissa", "Volos", "Chania"],
    "nigeria": ["Ibadan", "Enugu", "Abeokuta", "Kaduna", "Jos"],
    "kenya": ["Nakuru", "Kisumu", "Eldoret", "Nyeri", "Thika"],
    "ethiopia": ["Bahir Dar", "Gondar", "Mekelle", "Hawassa", "Jimma"],
    "south africa": ["Durban", "Bloemfontein", "Port Elizabeth", "Polokwane", "Nelspruit"],
    "united states": ["Portland", "Savannah", "Boise", "Madison", "Asheville"],
    "united kingdom": ["Sheffield", "Norwich", "Exeter", "Aberdeen", "York"],
    "canada": ["Halifax", "Victoria", "Saskatoon", "Kingston", "Sherbrooke"],
    "australia": ["Adelaide", "Hobart", "Geelong", "Townsville", "Ballarat"],
    "new zealand": ["Dunedin", "Napier", "Nelson", "Rotorua", "Invercargill"],
    "ireland": ["Galway", "Cork", "Limerick", "Kilkenny", "Sligo"],
}


def cities_of_country(country: str) -> List[str]:
    """Cities inside `country`; empty when the country is unknown."""
    return COUNTRY_CITIES.get((country or "").strip().lower(), [])

DEFAULT_REGION = "anglophone"


def region_of(nationality: str) -> str:
    key = (nationality or "").strip().lower()
    if key in NATIONALITY_TO_REGION:
        return NATIONALITY_TO_REGION[key]
    for nat, region in NATIONALITY_TO_REGION.items():
        if nat in key or key in nat:
            return region
    return DEFAULT_REGION
