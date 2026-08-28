function copyBibTeX() {
  const bibtexElement = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  const copyText = button?.querySelector('.copy-text');

  if (!bibtexElement || !button || !copyText) return;

  navigator.clipboard.writeText(bibtexElement.textContent).then(() => {
    button.classList.add('copied');
    copyText.textContent = 'Copied';
    setTimeout(() => {
      button.classList.remove('copied');
      copyText.textContent = 'Copy';
    }, 2000);
  }).catch(() => {
    const textArea = document.createElement('textarea');
    textArea.value = bibtexElement.textContent;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
  });
}
