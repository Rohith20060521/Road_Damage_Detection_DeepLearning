import requests

GEO_LOC_URL = "https://raw.githubusercontent.com/pradt2/always-online-stun/master/geoip_cache.txt"
IPV4_URL = "https://raw.githubusercontent.com/pradt2/always-online-stun/master/valid_ipv4s.txt"
GEO_USER_URL = "https://geolocation-db.com/json"

def getSTUNServer():
    try:
        # Fetch geoLocs data
        response = requests.get(GEO_LOC_URL, timeout=5)
        geoLocs = response.json()

        # Fetch latitude and longitude
        try:
            response = requests.get(GEO_USER_URL, timeout=5)
            user_data = response.json()
            latitude, longitude = user_data["latitude"], user_data["longitude"]
        except Exception:
            # Fallback to ip-api
            response = requests.get("http://ip-api.com/json", timeout=5)
            user_data = response.json()
            latitude, longitude = user_data["lat"], user_data["lon"]

        # Fetch and process IPV4 data
        response = requests.get(IPV4_URL, timeout=5)
        ip_addresses = response.text.strip().split('\n')

        # Find the closest STUN server
        def calculate_distance(addr):
            stunLat, stunLon = geoLocs.get(addr.split(':')[0], (0, 0))
            dist = ((latitude - stunLat) ** 2 + (longitude - stunLon) ** 2) ** 0.5
            return addr, dist

        closest_addr, _ = min(map(calculate_distance, ip_addresses), key=lambda x: x[1])

        # print("Free STUN Server :", closest_addr)  # prints the IP:PORT of the closest STUN server

        return closest_addr
    except Exception as e:
        print(f"Error fetching STUN server: {e}")
        return "stun.l.google.com:19302"

# getStunServer()