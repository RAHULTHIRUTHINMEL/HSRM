from dv import NetworkEventInput


#events connects events to eventserver
with NetworkEventInput(address='127.0.0.1', port=53093) as i:
    for event in i:
        print(event.timestamp)