
const uploadButton = document.getElementById('uploadButton');
const filePathElement = document.getElementById('filePath');

if (!uploadButton) {
  throw new Error('There is no button with id "uploadButton"');
}

uploadButton.addEventListener('click', async () => {
  const filePath = await window.electronAPI.openFile()
  console.log(filePath);
  filePathElement.textContent = filePath;
});