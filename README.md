# BotParking_KandEx
Kandidatarbete inom RL om att parkera en non-holonomic vehicle.


Bygger på en forskningsartikel om en closed-loop control solution för att flytta en non-holonomic vehicle (tex en enhjuling) från A till B där den också antar en specifik orientering i slutpositionen.

Koden är inte en slutlig lösning på problemet då jag temporärt har fortsatt arbetet i en notebook-fil på google colab och bytte till att använda ett biblioteket som heter StableBaselines3.

I implementationen finns en virtuell miljö, en implementering av en DDPG algoritm med ett neuralt nätverk som policy, samt ett antal olika rewardfunktioner, kod för visualisering och en träningsloop. De mer intressanta filerna är 
  - DDPG
  - SystemManager
  - SystemDynamics/State
  - Rewards
  - SimulationMain (scriptet som kör allt)
