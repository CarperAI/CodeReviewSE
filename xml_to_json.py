import xmltodict
import json

src = ["Badges","Comments","PostHistory","PostLinks","Posts","Tags","Users","Votes"]

for s in src:
    with open(s+".xml") as xml_file:
        data_dict = xmltodict.parse(xml_file.read())

    json_data = json.dumps(data_dict)

    with open(s+".json", "w") as json_file:
        json_file.write(json_data)
