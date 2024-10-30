function showPanel(panelId) {
    // Hide all panels
    document.querySelectorAll('.visualization-panel').forEach(function(panel) {
        panel.classList.remove('active-panel');
        panel.classList.add('hidden');
    });

    // Show the selected panel
    const activePanel = document.getElementById(panelId);
    if (activePanel) {
        activePanel.classList.add('active-panel');
        activePanel.classList.remove('hidden');
    }
}

// Function to make the clicked option active and deactivate others
function makeActive(activeOptionId) {
    // Deactivate all options
    document.querySelectorAll('.option-button').forEach(function(option) {
        option.classList.remove('active');
    });

    // Activate the clicked option
    const activeOption = document.getElementById(activeOptionId);
    if (activeOption) {
        activeOption.classList.add('active');
    }
    activeOption.classList.add('active');
    document.getElementById('panel').classList.add('active-panel');
}

// Event listeners for main option buttons
document.getElementById('dataInsightsOption').addEventListener('click', function() {
    makeActive('dataInsightsOption');
    showPanel('panel');
    document.getElementById('visualizationButtons').classList.remove('hidden');
});

document.getElementById('interactiveDashboardOption').addEventListener('click', function() {
    makeActive('interactiveDashboardOption');
    showPanel('interactiveDashboardContent');
    document.getElementById('visualizationButtons').classList.add('hidden');
});

document.getElementById('imdbPredictorOption').addEventListener('click', function() {
    makeActive('imdbPredictorOption');
    showPanel('imdbPredictorContent');
    document.getElementById('visualizationButtons').classList.add('hidden');
});

// Optional: Remove the video overlay after a certain time
setTimeout(function() {
    document.getElementById('introOverlay').style.display = 'none';
}, 4000); // Adjust time as needed

// Visualization buttons event listeners
var buttons = document.querySelectorAll('.visualization-button');
var panels = document.querySelectorAll('.visualization-panel');

buttons.forEach(function(btn) {
    btn.addEventListener('click', function() {
        // Remove active class from all buttons and panels
        buttons.forEach(function(button) {
            button.classList.remove('active');
        });
        panels.forEach(function(panel) {
            panel.classList.remove('active-panel');
            panel.classList.add('hidden');
        });

        // Add active class to clicked button and corresponding panel
        btn.classList.add('active');
        var panelId = btn.getAttribute('id').replace('btn', 'panel');
        document.getElementById(panelId).classList.add('active-panel');
        document.getElementById(panelId).classList.remove('hidden');
    });
});

var slideIndex = 1;
showSlides(slideIndex);

function moveSlide(n) {
  showSlides(slideIndex += n);
}

function showSlides(n) {
  var i;
  var slides = document.getElementsByClassName("carousel-slide");
  if (n > slides.length) { slideIndex = 1 }
  if (n < 1) { slideIndex = slides.length }
  for (i = 0; i < slides.length; i++) {
      slides[i].style.display = "none";
  }
  slides[slideIndex-1].style.display = "block";
}


document.getElementById('checkButton').addEventListener('click', function() {
    var type = document.getElementById('typeInput').value;
    var director = document.getElementById('directorInput').value;
    var country = document.getElementById('countryInput').value;
    var genre = document.getElementById('genreInput').value;

    fetch('http://localhost:8000/', {
        method: 'POST',
        body: JSON.stringify({ type: type, director: director, country: country, genre: genre }),
        headers: {
            'Content-Type': 'application/json'
        },
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('predictionResult').innerHTML = `Predicted IMDb Score : ${parseFloat(data.prediction).toFixed(4)}`;
    })
    .catch((error) => {
        console.error('Error:', error);
        document.getElementById('predictionResult').innerHTML = "Error fetching prediction.";
    });
});