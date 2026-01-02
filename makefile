# Variables
MESSAGE = "Mise à jour simulation Ligue 1"

# La commande par défaut
all: save

# Automatisation complète : add + commit + push
save:
	git add .
	git commit -m "Mise à jour simulation Ligue 1"
	git push origin main

# Version où tu peux choisir ton message (ex: make commit MSG="correction bug")
commit:
	git add .
	git commit -m "maj"
	git push origin main 