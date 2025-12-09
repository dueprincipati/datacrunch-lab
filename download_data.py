import os
import sys

def download(api_key, download_path="data"):
    print("Inizializzazione download dati...")
    
    # Assicura che la directory esista
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    print(f"Scaricando i dati in {download_path}...")
    
    # SINTASSI CORRETTA:
    # crunch setup [COMPETITION] [PROJECT_NAME] [DIRECTORY] --token [TOKEN] --size large
    # competition: datacrunch
    # project: submission (nome arbitrario per il setup locale)
    # directory: path dove scaricare i dati
    command = f"crunch setup datacrunch submission {download_path} --token {api_key} --size large --force"
    
    # Esegue il comando
    exit_code = os.system(command)
    
    if exit_code == 0:
        print("\n✅ Download completato con successo.")
    else:
        print("\n❌ Errore durante il download.")

if __name__ == "__main__":
    # Gestione input API Key (Priorità: Env Var -> Argomento -> Input manuale)
    API_KEY = os.getenv("CRUNCHDAO_API_KEY") 
    if not API_KEY:
        if len(sys.argv) > 1:
            API_KEY = sys.argv[1]
        else:
            API_KEY = input("Inserisci la tua (NUOVA) API Key CrunchDAO: ")
    
    download(API_KEY)