# Imbalanced-data
 A financial fraud use case. Work in progress!

## Introduzione

In questa repository esploriamo alcune tecniche per affrontare il problema della classificazione in situazioni di forte sbilancilamento nei dati. Il progetto è suddiviso in tre macro passaggi:

1. analisi descrittiva delle variabili;
2. classificazione supervisionata;
3. classificazione non supervisionata.

## Dati

Il dataset considerato è un insieme di 284.807 transazioni di carte di credito eseguite in Europa nel settembre del 2013 e copre un intervallo temporale di 2 giorni. 

Ogni transazione è innazitutto etichettata come regolare (**0**) oppure come frode (**1**). Il forte sbilanciamento fra le due classi dipende dal fatto che solo 492 transazioni sono di tipo frode.

Ogni transazione è poi descritta da una variabile *Time* (in secondi) che rappresenta l'intervallo temporale intercorso fra essa e la prima transazione del dataset, dall'importo scambiato o *Amount*  e da altre 29 variabili numeriche che per questione di privacy sono state opportunamente anonimizzate e trasformate tramite PCA. 

Il dataset e ulteriori informazioni posso essere trovati al seguente link: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Analisi descrittiva

In questa fase sono state fatte delle valutazioni di carattere generale sulle variabili che vanno a comporre il dataset. In particolare sono state confrontate per le due classi le distrubioni dei valori di ogni singolo descrittore. Il risultato è visualizzabile con l'aiuto dei grafici *var_dist_xxx.png* della cartella *plots*. E' stato inoltre verificato che le variabili hanno subito effetivamente una trasformazione PCA verificando la completa scorrelazione tramite la matrice di covarianza di cui sono riportati i valori in  *covariance_matrix.png*, sempre all'interno della cartella *plots*.

Non sono stati fatti ragionamenti qualitativi su questi risultati a causa dell'anonimizzazione dei descrittori che ha come conseguenza la perdita di significato nel contesto reale. Tuttavia a valle della classificazione potranno fornire alcune utili indicazioni.



## Classificazione supervisionata

### Approccio naïf 
Come primo passo sono stati addestrati due semplici modelli di *classificazione lineare* e *regressione logistica* per stabilire un *benchmark* iniziale sulle prestazioni e avere una prima idea di come il forte sbilanciamento impatti la soluzione. La performance ottenuta è rissumibile dal valore ROC-AUC che mediamente si attesta su un valore pari a 0.78. 

### Ribilanciamento del dataset

In condizione di forte sbilancamento nei dati, l'approccio standard per la classificazione prevede un sovra-campionamento della classe minoritaria e un sotto-campionamento della classe maggioritaria.

Se il sotto-campionamento non nasconde grosse insidie, il sovra-campionamento, eseguito con la tecnica SMOTE, è giustificabile se si assume che, sebbene numericamente ristretta, la popolazione della classe minoritaria è una buona fotografia del femeno sia in termini di caratteristiche che di variabilità. In poche parole la tecnica SMOTE genera nuovi dati a partire da un sottoinsieme, nel nostro caso frodi, mediandone i valori.

Osserviamo che le due operazioni non sono commutative ed è importante eseguire prima il sovra-campionamento per non perdere nessuna informazione sulla classe di cui si dispone già di poche istanze.

### Modelli di ensamble

Dopo la fase di ribilanciamento sono addestrati altri due tipi di modelli: una *random forest* e un *xgboost*. All'interno del codice è stata prevista una funzione per eseguire una *grid-search* di numerosi parametri di entrambi i modelli. Questa funzione restituisce le varie prestazioni di tutti i modelli provati e fra questi sceglie in automatico quello migliore basandosi sul valore ROC-AUC. Sulla base di questa scelta viene riaddestrato da capo il modello per estrarre tutte le caratteristiche di interesse: il grafico della curva ROC, la lista delle variabili più importanti nel processo decisionale e la matrice di confusione. 

Queste informazioni possono essere consultate nei rispettivi grafici all'interno della cartella *plots*. 

Le performace ottenute dal modello migliore, quasi sempre xgboost, si attestano mediamente intorno ad un valore di ROC-AUC di 0.93.

### Discussione dei risultati e valore di business

Analizziamo i risultati ottenuti in fase di test dal modello migliore. 

Il dataset di test conteneva 85.127 transazioni, di cui 158 frodi. Il modello è stato capace di individuare correttamente 134 frodi (84%) segnalando allo stesso tempo 345 falsi positivi.

Da questi dati emerge come gli agenti incaricati dell'analisi del transanto dovranno processare ora solo 479 transazioni anzichè 85.127 e cioè lo 0,006% dei casi. In questa piccola percentuale mediamente saranno contenute l'84% delle frodi totali con un tasso di una frode ogni quattro transazioni analizzate (il tasso precedente era una ogni duemila).

Infine consideriamo il grafico relativo all'importanza delle variabili nel processo decisionale. Questa visualizzazione restituisce soltanto un elenco ordinato di *feature* e dà nessuna informazione di tipo qualitativo. Tale informazione però può essere recuperata sfuttando il lavoro fatto durante l'analisi descrittiva. Per esempio la distribuzione della variabile **V4**, che il modello individua essere quella più importante per discernere fra le due classi, suggerisce che nel sottoinsieme delle frodi (istogramma rosso nell'immagine) il valore della variabile stessa è mediamente più alto rispetto a quello dell'insieme delle transazioni regolari (in blu).

Questo tipo di indicazione è particolarmente utile quando il numero di predittori  è molto grande e sopratutto quando le variabili considerate sono parlanti e possono effettivamente dare un indicazione concreta.



## Classificazione non supervisionata

### Idea

Con questo approccio **non** vengono forniti ai modelli le etichette che distinguono transazioni normali da transazioni fraudolente. L'unica informazione passata è una stima del rapporto che ci si aspetta fra le due classi che in questo contesto prende il nome di *contaminazione* e per il nostro caso sarà pari a 0.002. 

L'idea dietro questo tipo di approccio è sfruttare lo sbilanciamento dei dati a proprio favore e rilevare le frodi come anomalie di un processo che mediamente è rappresentato da transazioni regolari.  

Premettiamo che il dataset è stato sottocampionato perché l'addestramento di questi modelli richiede più risorse rispetto alle precedenti soluzioni. Tuttavia lo sbilanciamento del "nuovo" dataset è pressoché identico al precedente perché effettuato in maniera completamente casuale.

### Modelli

Sono stati addrestrati tre modelli che si basano su logiche differenti.

1. Isolation Forest: ensamble
2. Autoencoder: rete neurale.
3. Local Outlier Factor: densità.

Scegliere modelli che seguono varie logiche per il rilevamento di anomalie è utile perché ci permette di apprezzare il fenomeno da punti di vista leggermente diversi e in fase di output prenderemo in considerazione tutte le logiche applicate.

Questi algoritmi associano ad ogni istanza un valore di anomalia, più è alto questo numero più l'istanza è considerata anomala. Successivamente, in base alla contaminazione prevista, associano ad ogni osservazione l'etichetta. Nel caso di assenza di questo parametro è possibile sceglire una soglia a priori sopra la quale le istanze sono considerate anomalie.

Nella cartella *plots*, per ogni modello, sono riportati i valori di anomalia associati a ogni osservazione. I grafici in questione ci aiutano fra l'altro a capire quanto i modelli stanno effettivamente interpretando bene il fenomeno.

Sfruttando le etichette solo in un secondo momento, dopo la fase di training, possiamo dare una stima delle prestazioni e analizzare i risultati ottenuti. Le performace ottenute dal modello migliore, quasi sempre l'autoencoder, si attestano mediamente intorno ad un valore di ROC-AUC di 0.84.

### Discussione dei risultati e valore di business

Analizziamo i risultati ottenuti dall'unione dei tre modelli.

Il dataset campionato conteneva 51.070 transazioni, di cui 103 frodi. Il modello è stato capace di individuare correttamente 88 frodi (85%) segnalando allo stesso tempo 845 falsi positivi.

Da questi dati emerge come gli agenti incaricati dell'analisi del transanto dovranno processare ora solo 942 transazioni anzichè 51.070 e cioè lo 0,02% dei casi. In questa percentuale mediamente saranno contenute l'85% delle frodi totali con un tasso di una frode ogni dieci transazioni analizzate (il tasso precedente era una ogni duemila).

Se confrontato con il modello supervisionato, le prestazioni ottenute da questo approccio sono leggermente inferiori, ma ricordiamoci che gli algoritmi non supervisionati non utilizzano le etichette in fase di training!

Il non utilizzare le etichette non è necessariamente un speco di informazione anche qualora si abbiano a disposizione. Infatti se queste sono frutto di lavoro manuale da parte di qualche incaricato soffrono inevitabilmente del bias di chi le assegna. Questo nell'ambito frodi su carta di credito può non rappresentare un grosso problema essendo l'etichettatura certificata da chi subisce la frode, ma in altri ambiti, come l'antiricilaggio, certamente chi opera transazioni con l'obbiettivo di ripulire denaro presumibilmente non si denuncerà alle autorità!