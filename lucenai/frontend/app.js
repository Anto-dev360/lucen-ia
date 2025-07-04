// Données des cryptomonnaies
const cryptoData = {
  "BTC": {
    "nom": "Bitcoin",
    "symbole": "BTC",
    "prix_actuel": 105145.4,
    "market_cap": "2.09T",
    "volume_24h": "68.15B",
    "variation_24h": 3.43,
    "variation_7j": 1.4,
    "rang_marche": 1,
    "supply_circulant": "19.88M",
    "supply_max": "21M",
    "fondamentaux": {
      "technologie": "Proof of Work, Blockchain",
      "cas_usage": "Réserve de valeur, Moyen de paiement",
      "avantages": ["Première crypto", "Sécurité élevée", "Adoption massive"],
      "defis": ["Scalabilité", "Consommation énergétique"]
    },
    "onchain": {
      "adresses_actives": "1M+",
      "hash_rate": "700+ EH/s",
      "frais_moyens": "$15-50",
      "temps_confirmation": "10 min"
    }
  },
  "ETH": {
    "nom": "Ethereum",
    "symbole": "ETH",
    "prix_actuel": 2404.68,
    "market_cap": "290.29B",
    "volume_24h": "26.15B",
    "variation_24h": 6.48,
    "variation_7j": 7.58,
    "rang_marche": 2,
    "supply_circulant": "120.1M",
    "supply_max": "∞",
    "fondamentaux": {
      "technologie": "Proof of Stake, Smart Contracts",
      "cas_usage": "Plateforme dApps, DeFi, NFTs",
      "avantages": ["Écosystème développé", "Smart contracts", "Transition PoS"],
      "defis": ["Frais de gas", "Concurrence Layer 1"]
    },
    "onchain": {
      "adresses_actives": "500K+",
      "transactions_jour": "1.2M",
      "frais_moyens": "$3-20",
      "temps_confirmation": "12 sec"
    }
  },
  "SOL": {
    "nom": "Solana",
    "symbole": "SOL",
    "prix_actuel": 144.268,
    "market_cap": "76.66B",
    "volume_24h": "5.95B",
    "variation_24h": 7.65,
    "variation_7j": 7.99,
    "rang_marche": 6,
    "supply_circulant": "531M",
    "supply_max": "∞",
    "fondamentaux": {
      "technologie": "Proof of History, Scalabilité haute",
      "cas_usage": "DeFi, NFTs, Gaming, Paiements rapides",
      "avantages": ["Vitesse élevée", "Frais bas", "Écosystème en croissance"],
      "defis": ["Pannes réseau", "Centralisation relative"]
    },
    "onchain": {
      "adresses_actives": "300K+",
      "transactions_jour": "50M+",
      "frais_moyens": "$0.001",
      "temps_confirmation": "0.4 sec"
    }
  },
  "TAO": {
    "nom": "Bittensor",
    "symbole": "TAO",
    "prix_actuel": 280.0,
    "market_cap": "2.1B",
    "volume_24h": "150M",
    "variation_24h": 3.49,
    "variation_7j": 5.2,
    "rang_marche": 48,
    "supply_circulant": "7.5M",
    "supply_max": "21M",
    "fondamentaux": {
      "technologie": "IA décentralisée, Réseaux de neurones",
      "cas_usage": "Intelligence artificielle, Machine learning",
      "avantages": ["Secteur IA en croissance", "Modèle unique", "Innovation"],
      "defis": ["Complexité technique", "Adoption limitée"]
    },
    "onchain": {
      "adresses_actives": "5K+",
      "validateurs": "1000+",
      "frais_moyens": "$0.1",
      "temps_confirmation": "12 sec"
    }
  },
  "LINK": {
    "nom": "Chainlink",
    "symbole": "LINK",
    "prix_actuel": 13.05,
    "market_cap": "8.85B",
    "volume_24h": "619.88M",
    "variation_24h": 10.02,
    "variation_7j": 11.23,
    "rang_marche": 14,
    "supply_circulant": "626M",
    "supply_max": "1B",
    "fondamentaux": {
      "technologie": "Oracle décentralisé, Intégrations cross-chain",
      "cas_usage": "Oracles, Données externe, Interopérabilité",
      "avantages": ["Leader oracles", "Partenariats solides", "Utilité réelle"],
      "defis": ["Concurrence", "Dépendance intégrations"]
    },
    "onchain": {
      "adresses_actives": "100K+",
      "oracles_actifs": "1000+",
      "frais_moyens": "$5-15",
      "temps_confirmation": "12 sec"
    }
  },
  "NEAR": {
    "nom": "NEAR Protocol",
    "symbole": "NEAR",
    "prix_actuel": 2.14,
    "market_cap": "2.63B",
    "volume_24h": "235.77M",
    "variation_24h": 12.03,
    "variation_7j": 1.9,
    "rang_marche": 36,
    "supply_circulant": "1.22B",
    "supply_max": "∞",
    "fondamentaux": {
      "technologie": "Sharding, PoS, IA native",
      "cas_usage": "dApps, IA, Chain abstraction",
      "avantages": ["Sharding efficace", "Développement facile", "Focus IA"],
      "defis": ["Concurrence Layer 1", "Adoption développeurs"]
    },
    "onchain": {
      "adresses_actives": "50K+",
      "transactions_jour": "500K+",
      "frais_moyens": "$0.001",
      "temps_confirmation": "2 sec"
    }
  },
  "ONDO": {
    "nom": "Ondo Finance",
    "symbole": "ONDO",
    "prix_actuel": 1.85,
    "market_cap": "2.6B",
    "volume_24h": "180M",
    "variation_24h": 5.2,
    "variation_7j": 8.1,
    "rang_marche": 37,
    "supply_circulant": "1.4B",
    "supply_max": "10B",
    "fondamentaux": {
      "technologie": "Tokenisation d'actifs réels, DeFi-TradFi",
      "cas_usage": "RWA, Obligations tokenisées, Yield",
      "avantages": ["Marché RWA en croissance", "Produits institutionnels", "Conformité réglementaire"],
      "defis": ["Réglementation", "Adoption institutionnelle lente"]
    },
    "onchain": {
      "adresses_actives": "10K+",
      "tvl": "$500M+",
      "frais_moyens": "$2-8",
      "temps_confirmation": "12 sec"
    }
  },
  "KAS": {
    "nom": "Kaspa",
    "symbole": "KAS",
    "prix_actuel": 0.12,
    "market_cap": "3.0B",
    "volume_24h": "45M",
    "variation_24h": 2.1,
    "variation_7j": 4.3,
    "rang_marche": 25,
    "supply_circulant": "25B",
    "supply_max": "28.7B",
    "fondamentaux": {
      "technologie": "GHOSTDAG, BlockDAG, PoW",
      "cas_usage": "Paiements rapides, Transactions parallèles",
      "avantages": ["Innovation technique", "Vitesse élevée", "Sécurité PoW"],
      "defis": ["Reconnaissance limitée", "Écosystème naissant"]
    },
    "onchain": {
      "adresses_actives": "20K+",
      "blocs_par_seconde": "1-10",
      "frais_moyens": "$0.0001",
      "temps_confirmation": "1 sec"
    }
  },
  "RENDER": {
    "nom": "Render Network",
    "symbole": "RENDER",
    "prix_actuel": 2.85,
    "market_cap": "1.5B",
    "volume_24h": "85M",
    "variation_24h": -1.89,
    "variation_7j": 2.1,
    "rang_marche": 55,
    "supply_circulant": "526M",
    "supply_max": "536M",
    "fondamentaux": {
      "technologie": "GPU computing, DePIN, Rendu distribué",
      "cas_usage": "Rendu 3D, IA, Computing décentralisé",
      "avantages": ["Marché DePIN", "Utilité réelle", "Migration Solana"],
      "defis": ["Concurrence centralisée", "Adoption créateurs"]
    },
    "onchain": {
      "adresses_actives": "15K+",
      "gpu_nodes": "5000+",
      "frais_moyens": "$0.001",
      "temps_confirmation": "0.4 sec"
    }
  },
  "AAVE": {
    "nom": "Aave",
    "symbole": "AAVE",
    "prix_actuel": 185.0,
    "market_cap": "2.8B",
    "volume_24h": "125M",
    "variation_24h": 4.2,
    "variation_7j": 6.8,
    "rang_marche": 42,
    "supply_circulant": "15.1M",
    "supply_max": "16M",
    "fondamentaux": {
      "technologie": "Protocole de prêt, Liquidité, DeFi",
      "cas_usage": "Prêts/emprunts, Yield farming, Liquidité",
      "avantages": ["Leader DeFi", "TVL élevée", "Innovation continue"],
      "defis": ["Réglementation DeFi", "Concurrence protocoles"]
    },
    "onchain": {
      "adresses_actives": "50K+",
      "tvl": "$12B+",
      "frais_moyens": "$3-12",
      "temps_confirmation": "12 sec"
    }
  }
};

// Variables globales
let currentCrypto = 'BTC';
let priceChart = null;

// Système de calcul du score DYOR
function calculateDYORScore(crypto) {
  const data = cryptoData[crypto];
  
  // Performance de prix (20%)
  const priceScore = Math.max(0, Math.min(100, 
    (data.variation_24h + data.variation_7j) * 5 + 50
  ));
  
  // Solidité technologique (25%)
  const techScore = getTechScore(crypto);
  
  // Métriques onchain (25%)
  const onchainScore = getOnchainScore(crypto);
  
  // Position de marché (15%)
  const marketScore = Math.max(0, 100 - (data.rang_marche - 1) * 2);
  
  // Potentiel de croissance (15%)
  const growthScore = getGrowthScore(crypto);
  
  const totalScore = Math.round(
    (priceScore * 0.2) + 
    (techScore * 0.25) + 
    (onchainScore * 0.25) + 
    (marketScore * 0.15) + 
    (growthScore * 0.15)
  );
  
  return {
    total: totalScore,
    price: Math.round(priceScore),
    tech: Math.round(techScore),
    onchain: Math.round(onchainScore),
    market: Math.round(marketScore),
    growth: Math.round(growthScore)
  };
}

function getTechScore(crypto) {
  const techScores = {
    'BTC': 95, 'ETH': 90, 'SOL': 85, 'TAO': 80,
    'LINK': 88, 'NEAR': 82, 'ONDO': 75, 'KAS': 78,
    'RENDER': 83, 'AAVE': 87
  };
  return techScores[crypto] || 70;
}

function getOnchainScore(crypto) {
  const onchainScores = {
    'BTC': 90, 'ETH': 85, 'SOL': 88, 'TAO': 70,
    'LINK': 82, 'NEAR': 85, 'ONDO': 75, 'KAS': 80,
    'RENDER': 78, 'AAVE': 85
  };
  return onchainScores[crypto] || 70;
}

function getGrowthScore(crypto) {
  const growthScores = {
    'BTC': 65, 'ETH': 75, 'SOL': 85, 'TAO': 90,
    'LINK': 70, 'NEAR': 80, 'ONDO': 85, 'KAS': 75,
    'RENDER': 80, 'AAVE': 72
  };
  return growthScores[crypto] || 70;
}

// Formatage des prix
function formatPrice(price) {
  if (price >= 1000) {
    return '$' + price.toLocaleString('fr-FR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  } else {
    return '$' + price.toFixed(4);
  }
}

// Création des tags
function createTags(items, className = 'tag') {
  return items.map(item => `<span class="${className}">${item}</span>`).join('');
}

// Mise à jour de l'interface
function updateUI(crypto) {
  const data = cryptoData[crypto];
  const scores = calculateDYORScore(crypto);
  
  // Nom et score global
  document.getElementById('cryptoName').textContent = data.nom;
  document.getElementById('dyorScore').textContent = scores.total;
  document.getElementById('scoreGauge').style.width = scores.total + '%';
  
  // Rating
  const rating = document.getElementById('dyorRating');
  rating.className = 'status';
  if (scores.total >= 80) {
    rating.classList.add('status--success');
    rating.textContent = 'Excellent';
  } else if (scores.total >= 60) {
    rating.classList.add('status--info');
    rating.textContent = 'Bon';
  } else if (scores.total >= 40) {
    rating.classList.add('status--warning');
    rating.textContent = 'Moyen';
  } else {
    rating.classList.add('status--error');
    rating.textContent = 'Risqué';
  }
  
  // Vue d'ensemble
  document.getElementById('currentPrice').textContent = formatPrice(data.prix_actuel);
  document.getElementById('marketCap').textContent = '$' + data.market_cap;
  document.getElementById('volume24h').textContent = '$' + data.volume_24h;
  document.getElementById('marketRank').textContent = '#' + data.rang_marche;
  
  // Variations
  const var24h = document.getElementById('variation24h');
  const var7d = document.getElementById('variation7d');
  
  var24h.textContent = (data.variation_24h > 0 ? '+' : '') + data.variation_24h.toFixed(2) + '%';
  var24h.className = 'variation-value ' + (data.variation_24h > 0 ? 'positive' : 'negative');
  
  var7d.textContent = (data.variation_7j > 0 ? '+' : '') + data.variation_7j.toFixed(2) + '%';
  var7d.className = 'variation-value ' + (data.variation_7j > 0 ? 'positive' : 'negative');
  
  // Fondamentaux
  document.getElementById('technology').textContent = data.fondamentaux.technologie;
  document.getElementById('useCase').textContent = data.fondamentaux.cas_usage;
  document.getElementById('advantages').innerHTML = createTags(data.fondamentaux.avantages);
  document.getElementById('challenges').innerHTML = createTags(data.fondamentaux.defis, 'tag');
  document.getElementById('supplyCirculating').textContent = data.supply_circulant;
  document.getElementById('supplyMax').textContent = data.supply_max;
  
  // Onchain métriques
  document.getElementById('activeAddresses').textContent = data.onchain.adresses_actives;
  document.getElementById('avgFees').textContent = data.onchain.frais_moyens;
  document.getElementById('confirmationTime').textContent = data.onchain.temps_confirmation;
  
  // Métrique spéciale selon la crypto
  const specialLabel = document.getElementById('specialMetricLabel');
  const specialValue = document.getElementById('specialMetricValue');
  
  if (crypto === 'BTC') {
    specialLabel.textContent = 'Hash Rate';
    specialValue.textContent = data.onchain.hash_rate;
  } else if (crypto === 'ETH') {
    specialLabel.textContent = 'Transactions/jour';
    specialValue.textContent = data.onchain.transactions_jour;
  } else if (crypto === 'SOL') {
    specialLabel.textContent = 'Transactions/jour';
    specialValue.textContent = data.onchain.transactions_jour;
  } else if (crypto === 'TAO') {
    specialLabel.textContent = 'Validateurs';
    specialValue.textContent = data.onchain.validateurs;
  } else if (crypto === 'LINK') {
    specialLabel.textContent = 'Oracles actifs';
    specialValue.textContent = data.onchain.oracles_actifs;
  } else if (crypto === 'NEAR') {
    specialLabel.textContent = 'Transactions/jour';
    specialValue.textContent = data.onchain.transactions_jour;
  } else if (crypto === 'ONDO') {
    specialLabel.textContent = 'TVL';
    specialValue.textContent = data.onchain.tvl;
  } else if (crypto === 'KAS') {
    specialLabel.textContent = 'Blocs/seconde';
    specialValue.textContent = data.onchain.blocs_par_seconde;
  } else if (crypto === 'RENDER') {
    specialLabel.textContent = 'GPU Nodes';
    specialValue.textContent = data.onchain.gpu_nodes;
  } else if (crypto === 'AAVE') {
    specialLabel.textContent = 'TVL';
    specialValue.textContent = data.onchain.tvl;
  }
  
  // Santé du réseau
  const networkHealth = document.getElementById('networkHealth');
  networkHealth.className = 'status status--success';
  networkHealth.textContent = 'Excellent';
  
  // Scores détaillés
  updateScoreBar('priceScore', scores.price);
  updateScoreBar('techScore', scores.tech);
  updateScoreBar('onchainScore', scores.onchain);
  updateScoreBar('marketScore', scores.market);
  updateScoreBar('growthScore', scores.growth);
  
  
  // Affichage Feedback on X
  const feedbackSection = document.getElementById('dyor_feelings');
  if (crypto === 'BTC') {
    feedbackSection.style.display = 'block';
  } else {
    feedbackSection.style.display = 'none';
  }

  // Recommandations
  updateRecommendations(crypto, scores);
  
  // Graphique
  updatePriceChart(crypto);
}

function updateScoreBar(id, score) {
  const bar = document.getElementById(id);
  const valueEl = bar.parentElement.nextElementSibling;
  bar.style.width = score + '%';
  valueEl.textContent = score + '%';
}

function updateRecommendations(crypto, scores) {
  const recommendations = document.getElementById('recommendations');
  let recList = [];
  
  if (scores.total >= 80) {
    recList.push('<div class="recommendation-item status--success">✅ Actif recommandé pour investissement</div>');
  } else if (scores.total >= 60) {
    recList.push('<div class="recommendation-item status--info">ℹ️ Actif intéressant à surveiller</div>');
  } else {
    recList.push('<div class="recommendation-item status--warning">⚠️ Analyser les risques avant investissement</div>');
  }
  
  if (scores.price < 50) {
    recList.push('<div class="recommendation-item status--warning">⚠️ Performance récente faible</div>');
  }
  
  if (scores.growth > 80) {
    recList.push('<div class="recommendation-item status--success">🚀 Fort potentiel de croissance</div>');
  }
  
  recommendations.innerHTML = recList.join('');
}

function updatePriceChart(crypto) {
  const ctx = document.getElementById('priceChart').getContext('2d');
  
  if (priceChart) {
    priceChart.destroy();
  }
  
  // Données simulées pour le graphique
  const data = cryptoData[crypto];
  const basePrice = data.prix_actuel;
  const labels = ['6j', '5j', '4j', '3j', '2j', '1j', 'Aujourd\'hui'];
  const prices = generatePriceHistory(basePrice, data.variation_7j);
  
  priceChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Prix',
        data: prices,
        borderColor: '#1FB8CD',
        backgroundColor: 'rgba(31, 184, 205, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: false,
          grid: {
            color: 'rgba(94, 82, 64, 0.1)'
          },
          ticks: {
            callback: function(value) {
              return formatPrice(value);
            }
          }
        },
        x: {
          grid: {
            color: 'rgba(94, 82, 64, 0.1)'
          }
        }
      }
    }
  });
}

function generatePriceHistory(currentPrice, variation7d) {
  const prices = [];
  const dailyVariation = variation7d / 7;
  
  for (let i = 6; i >= 0; i--) {
    const variance = (Math.random() - 0.5) * 0.1; // ±5% de variance
    const price = currentPrice * (1 - (dailyVariation * i / 100) + variance);
    prices.push(price);
  }
  
  return prices;
}

// Gestion du thème
function toggleTheme() {
  const body = document.body;
  const themeIcon = document.querySelector('.theme-icon');
  
  if (body.hasAttribute('data-color-scheme')) {
    const currentTheme = body.getAttribute('data-color-scheme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    body.setAttribute('data-color-scheme', newTheme);
    themeIcon.textContent = newTheme === 'dark' ? '☀️' : '🌙';
  } else {
    body.setAttribute('data-color-scheme', 'dark');
    themeIcon.textContent = '☀️';
  }
}

// Initialisation
document.addEventListener('DOMContentLoaded', function() {
  // Sélecteur de crypto
  const cryptoSelect = document.getElementById('cryptoSelect');
  cryptoSelect.addEventListener('change', function() {
    currentCrypto = this.value;
    updateUI(currentCrypto);
  });
  
  // Toggle de thème
  const themeToggle = document.getElementById('themeToggle');
  themeToggle.addEventListener('click', toggleTheme);
  
  // Initialisation avec BTC
  updateUI(currentCrypto);
});