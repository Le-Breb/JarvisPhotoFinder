# Interface Web de Labellisation des Visages

## Vue d'ensemble

L'onglet **Label** dans l'interface web permet de labelliser et valider les clusters de visages directement depuis le navigateur. Cette fonctionnalité est intégrée à l'application Next.js existante.

## Accès

Accédez à l'interface via :
- URL : `http://localhost:3000/label`
- Navigation : Cliquez sur l'onglet **Label** dans le header (à côté de Search et People)

## Fonctionnalités

### 1. Labelliser les Clusters (Onglet "Label Clusters")

**Objectif** : Assigner des noms aux clusters qui contiennent plus de 5 photos et qui n'ont pas encore de nom (ou qui ont le nom par défaut "Person X").

#### Comment ça marche :

1. **Chargement automatique** : Les clusters non nommés sont chargés et triés par nombre de photos (plus grand en premier)

2. **Affichage** :
   - Photo du visage représentatif (croppée avec padding)
   - Nombre de photos dans le cluster
   - Barre de progression
   - Pourcentage de complétion

3. **Actions disponibles** :
   - **Saisir un nom** : Entrez le nom de la personne dans le champ texte
   - **Submit Name** : Valider le nom (ou appuyez sur Entrée)
   - **Skip** : Passer ce cluster sans le nommer

4. **Sauvegarde** : Les changements sont sauvegardés immédiatement dans `clusters.json`

#### Interface :
```
┌────────────────────────────────────────────┐
│ Label Cluster 1 of 15      [67% Complete]  │
│ This cluster contains 23 photos            │
│ ─────────────────────────────────────────  │
│                                            │
│         [Photo du visage cropée]           │
│                                            │
│ Person Name: [________________]            │
│                                            │
│     [Submit Name]  [Skip]                  │
└────────────────────────────────────────────┘
```

### 2. Valider les Visages en Bordure (Onglet "Validate Faces")

**Objectif** : Vérifier et corriger les visages "incertains" qui sont proches de la frontière entre plusieurs clusters.

#### Critères d'incertitude :
Un visage est considéré comme incertain si :
- Distance au centroïde du cluster > 0.4
- OU distance à un autre cluster < 0.4

#### Comment ça marche :

1. **Chargement** :
   - Calcul des centroïdes de tous les clusters (via script Python backend)
   - Détection des visages en bordure
   - Chargement de la liste de tous les clusters existants

2. **Affichage** :
   - Photo du visage cropée
   - Nom du cluster suggéré
   - Distance au centroïde
   - Barre de progression

3. **Actions disponibles** :
   - **Yes, Correct** : Le visage est bien dans le bon cluster
   - **No, Incorrect** : Ouvre un dialogue pour réassigner le visage

4. **Réassignation** :
   - Liste de toutes les personnes existantes (avec photo et nombre de visages)
   - Option "Create New Person" pour créer un nouveau cluster
   - Cliquez sur la personne correcte pour réassigner le visage

#### Interface :
```
┌────────────────────────────────────────────┐
│ Validate Face 5 of 42      [12% Complete]  │
│ Distance to cluster: 0.456                 │
│ ─────────────────────────────────────────  │
│                                            │
│         [Photo du visage cropée]           │
│                                            │
│ ⚠ Is this person "John Doe"?              │
│                                            │
│  [✓ Yes, Correct]  [✗ No, Incorrect]      │
└────────────────────────────────────────────┘
```

## Architecture Technique

### Frontend (Next.js + React)

#### Pages :
- `/app/label/page.tsx` : Page principale avec les deux onglets

#### Composants :
- `/components/face-image.tsx` : Affichage des visages croppés avec padding
- `/components/ui/progress.tsx` : Barre de progression
- `/components/ui/badge.tsx` : Badges pour les statistiques
- `/components/ui/tabs.tsx` : Système d'onglets

#### Navigation :
- `/components/header.tsx` : Ajout du bouton "Label" dans la navigation

### Backend (Next.js API Routes)

#### Routes API créées :

1. **GET `/api/label/unlabeled`**
   - Retourne les clusters avec >5 photos sans nom
   - Triés par nombre de photos (décroissant)
   - Authentification requise

2. **GET `/api/label/boundary`**
   - Calcule les centroïdes des clusters (via Python)
   - Détecte les visages avec distance > seuil
   - Retourne les visages incertains avec métriques
   - Authentification requise

3. **POST `/api/label/reassign`**
   - Réassigne un visage d'un cluster à un autre
   - Supporte la création de nouveaux clusters
   - Nettoie les clusters vides
   - Authentification requise

4. **PUT `/api/people/[id]`** (existant)
   - Met à jour le nom d'un cluster
   - Utilisé pour la labellisation

### Traitement des Embeddings

La détection des visages en bordure utilise un script Python temporaire pour :
1. Charger les embeddings depuis `embeddings.npy`
2. Normaliser les vecteurs (L2)
3. Calculer les centroïdes de chaque cluster
4. Mesurer les distances pour chaque visage
5. Identifier les visages incertains

```python
# Critère d'incertitude
is_uncertain = (
    distance_to_centroid > 0.4 or
    min_distance_to_other < 0.4
)
```

## Workflow d'Utilisation Recommandé

### Première Utilisation

1. **Indexer les visages** (si pas déjà fait) :
   ```bash
   cd website/python
   source .venv/bin/activate
   python index_faces.py index
   python index_faces.py cluster
   ```

2. **Démarrer le serveur web** :
   ```bash
   cd website
   npm run dev
   ```

3. **Se connecter** :
   - URL : `http://localhost:3000`
   - Login avec vos identifiants

4. **Accéder à l'onglet Label** :
   - Cliquez sur "Label" dans la navigation
   - Choisissez "Label Clusters"

5. **Labelliser les principaux clusters** :
   - Nommez les personnes que vous reconnaissez
   - Skippez celles que vous ne reconnaissez pas

6. **Valider les visages incertains** :
   - Passez à l'onglet "Validate Faces"
   - Vérifiez les assignations suggérées
   - Corrigez les erreurs

### Utilisation Régulière

Après avoir ajouté de nouvelles photos :

1. **Ré-indexer** :
   ```bash
   cd website/python
   source .venv/bin/activate
   python index_faces.py index
   python index_faces.py cluster
   ```

2. **Ouvrir l'interface web** et aller sur `/label`

3. **Les nouveaux clusters non nommés apparaîtront automatiquement**

## Intégration avec les Autres Fonctionnalités

### Synchronisation en Temps Réel

Les changements effectués dans l'onglet Label sont **immédiatement visibles** :

1. **Page People (`/people`)** :
   - Les noms mis à jour apparaissent instantanément
   - Les clusters fusionnés sont reflétés
   - Les visages réassignés changent de personne

2. **Recherche (`/search`)** :
   - Recherche par nom fonctionne avec les nouveaux noms
   - Les résultats de recherche par visage sont mis à jour

3. **Détails de personne (`/people/[id]`)** :
   - Les visages réassignés apparaissent dans les bons profils
   - Les noms modifiés sont affichés

### Persistance des Données

Tous les changements sont sauvegardés dans `website/python/faces/clusters.json` :
- Format JSON lisible et éditable
- Pas de base de données nécessaire
- Backup automatique avant modifications importantes (selon les routes API)

## Avantages vs Application GUI Séparée

✅ **Intégré dans le workflow existant** : Pas besoin de lancer une application séparée

✅ **Accessible à distance** : Utilisable depuis n'importe quel navigateur

✅ **Authentification** : Sécurisé avec NextAuth

✅ **Interface moderne** : UI cohérente avec le reste de l'application

✅ **Responsive** : Fonctionne sur desktop, tablette et mobile

✅ **Pas de dépendances système** : Pas besoin d'installer Tkinter ou Qt

## Personnalisation

### Ajuster le Seuil de Détection

Pour modifier la sensibilité de détection des visages incertains, éditez `/app/api/label/boundary/route.ts:86` :

```python
distance_threshold = 0.4  # Valeur par défaut
```

**Recommandations** :
- `0.3` : Plus strict, détecte plus de visages incertains
- `0.4` : Équilibré (par défaut)
- `0.5` : Plus permissif, détecte moins de visages incertains

### Modifier le Seuil de Photos

Pour changer le nombre minimum de photos requis pour qu'un cluster soit proposé au labellisation, éditez `/app/api/label/unlabeled/route.ts:36` :

```typescript
if (isUnlabeled && faces.length > 5) {  // Changez 5 ici
```

## Dépannage

### Problème : "No unlabeled clusters found"

**Cause** : Tous les clusters sont déjà nommés ou ont moins de 5 photos.

**Solution** :
1. Vérifiez la page People pour voir les clusters existants
2. Ajustez le seuil de photos minimum si nécessaire
3. Ajoutez plus de photos et ré-indexez

### Problème : "No boundary faces found"

**Cause** : Tous les visages sont bien assignés (bon clustering).

**Solution** : C'est normal ! Cela signifie que le clustering est de bonne qualité.

### Problème : "Failed to load boundary faces"

**Cause** : Erreur lors de l'exécution du script Python.

**Solutions** :
1. Vérifiez que Python est accessible : `which python`
2. Vérifiez que les packages sont installés : `pip list | grep numpy`
3. Vérifiez les logs du serveur Next.js

### Problème : Les images ne s'affichent pas

**Cause** : Chemins d'images incorrects ou permissions.

**Solutions** :
1. Vérifiez que les images sont dans `website/python/images/`
2. Vérifiez que le serveur Next.js peut accéder aux fichiers
3. Consultez les logs du navigateur (F12 > Console)

## Performance

### Temps de Chargement

- **Clusters non nommés** : ~100-500ms (lecture JSON)
- **Visages en bordure** : ~2-5s (calcul des centroïdes + distances)

Le calcul des visages en bordure est plus lent car il nécessite :
1. Chargement des embeddings (12MB)
2. Normalisation L2 de tous les vecteurs
3. Calcul de 435 centroïdes
4. Calcul de ~6000 distances

### Optimisations Possibles

Pour améliorer les performances :

1. **Cacher les centroïdes** : Sauvegarder les centroïdes après clustering
2. **Pagination** : Charger les visages par lots de 50
3. **Web Workers** : Calculs côté client avec WebAssembly
4. **Index FAISS** : Utiliser FAISS pour la recherche de plus proches voisins

## Fichiers Modifiés/Créés

### Nouveaux Fichiers

```
website/
├── app/
│   ├── label/
│   │   └── page.tsx                          # Page principale
│   └── api/
│       └── label/
│           ├── unlabeled/route.ts            # API clusters non nommés
│           ├── boundary/route.ts             # API visages en bordure
│           └── reassign/route.ts             # API réassignation
├── components/
│   ├── face-image.tsx                        # Composant affichage visage
│   └── ui/
│       ├── progress.tsx                      # Barre de progression
│       ├── badge.tsx                         # Badge
│       └── tabs.tsx                          # Onglets
└── FACE_LABELING_WEB.md                      # Cette documentation
```

### Fichiers Modifiés

```
website/
├── components/
│   └── header.tsx                            # + Bouton Label
└── package.json                              # + Dépendances Radix UI
```

## Sécurité

### Authentification

Toutes les routes API requièrent une authentification via NextAuth :
```typescript
const session = await getServerSession(authOptions)
if (!session) {
  return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
}
```

### Validation des Entrées

- Validation des IDs de clusters
- Validation des noms (non vides)
- Vérification de l'existence des clusters source/cible
- Protection contre les suppressions accidentelles

### Permissions

Actuellement, tous les utilisateurs authentifiés peuvent labelliser. Pour restreindre aux admins uniquement, ajoutez :

```typescript
if ((session.user as any)?.role !== 'admin') {
  return NextResponse.json({ error: "Admin only" }, { status: 403 })
}
```

## Support

Pour plus d'informations :
- Documentation générale : `PEOPLE_FEATURE.md`
- Configuration initiale : `PEOPLE_SETUP.md`
- Guide rapide : `PEOPLE_QUICKSTART.md`

## Améliorations Futures

- [ ] Pagination pour les listes longues
- [ ] Affichage de plusieurs photos représentatives par cluster
- [ ] Recherche/filtre dans la liste des personnes
- [ ] Raccourcis clavier (flèches, Y/N)
- [ ] Mode batch pour labelliser plusieurs clusters d'un coup
- [ ] Visualisation du cluster entier avant assignation
- [ ] Undo/Redo
- [ ] Export des statistiques de labellisation
- [ ] Indicateur de confiance du clustering
- [ ] Suggestion de noms basée sur la similarité
