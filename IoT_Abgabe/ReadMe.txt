Dieses ReadMe beschreibt die Files die in diesem Ordner vorhanden sind. 

Unterordner "RaspPi Daten"

1. Ordner "Dataset" 			-> Enthält die Fotos die verwendet wurden um unsere Gesichter zu trainieren. 
2. Ordner "Images" 			-> Enthält die Fotos die gemacht werden wenn ein Gesicht erkennt wird oder eine unbekannt Person versucht das Login zu verwenden
3. File "climate_log.json 		-> Enthält die information, wieviel Personen sich in einem Raum befinden und ob die Klimasteuerung ein- /ausgeschaltet ist
4. File "encodings.pickle" 		-> Trainingsdatei der Gesichtserkennung
5. File "Face_room_train.py" 		-> Startet das Training der Gesichter inklusive Raumzuweisung
6. File "facial_req.py" 		-> Originales Gesichtserkennungs Script
7. File "facial_req_Room.py" 		-> Abgeändertes Gesichtserkennungsskript, welches aus Node-Red Durchgeführt wird.
8. File "headshot_picam.py" 		-> Code um fotos zu erstellen welche ins "Dataset" gespeichert werden
9. File "log.json" 			-> Log-Datei welche alle eintrittsversuche aufzeigt. 
10. File "state.json" 			-> Log-Datei welche aufzeigt ob eine Person eingeloggt ist und wieviele Personen pro Raum eingeloggt sind. 

Unterornder "Node-Red Flows"		-> Enthält das Json-File aller verwendeten Node-Red Flows

"Video_Dashboard.mkv" 			-> Zeigt die Dargestellten Dashboards
"Video_Gesichtserkennung.mkv"		-> Zeigt die Gesichtserkennung in Software
"Video_Gesichtserkennung_Pi.mov"	-> Zeigt die Gesichtserkennung auf dem Pi. 