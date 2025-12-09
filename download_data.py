import os
import crunch
import shutil

def download(api_key, download_path="data"):
    print("Inizializzazione download dati...")
    
    # Setup della directory
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    # Inizializza client
    client = crunch.Client(api_key=api_key)
    
    # Scarica i dati (X_train, y_train, X_test)
    # Nota: Questo scaricherà i file parquet nella cartella corrente o specificata
    print(f"Scaricando i dati in {download_path}...")
    
    # Esegue il comando di download della CLI tramite python
    # Assicurati di avere l'API key corretta dalla piattaforma
    os.system(f"crunch setup --token {api_key} --directory {download_path} --size large")
    
    print("Download completato.")

if __name__ == "__main__":
    # Prende la chiave dalle variabili d'ambiente o input manuale
    API_KEY = os.getenv("CRUNCHDAO_API_KEY") 
    if not API_KEY:
        API_KEY = input("Inserisci la tua API Key CrunchDAO: ")
    
    download(API_KEY)