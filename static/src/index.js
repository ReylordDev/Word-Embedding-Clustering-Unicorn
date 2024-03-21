document.addEventListener('DOMContentLoaded', (event) => {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const uploadForm = document.getElementById('uploadForm');

  dropZone.addEventListener("click", function () {
    fileInput.click();
  });

  fileInput.addEventListener("change", function () {
    if (fileInput.files.length > 0) {
      uploadForm.submit();
    }
  });

  dropZone.addEventListener("dragover", function (e) {
    this.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", function (e) {
    this.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", function (e) {
    e.preventDefault();
    e.stopPropagation();
    this.classList.remove("dragover");

    let file = e.dataTransfer.files[0];
    fileInput.files = e.dataTransfer.files;
    uploadForm.submit();
  });

  const runClusteringButton = document.getElementById('runClusteringButton');

  runClusteringButton.addEventListener("click", function () {
    fetch('/api/run_clustering', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({}),
    })
      .then(response => response.json())
      .then(data => {
        console.log('Success:', data);
        window.location.href = '/';
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  });



});
