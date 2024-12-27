document.getElementById('analysisForm').addEventListener('submit', async (e) => {
    e.preventDefault();
   
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
   
    loading.classList.remove('d-none');
    results.innerHTML = '';
   
    const data = {
        company1: document.getElementById('company1').value,
        company2: document.getElementById('company2').value,
        startDate: document.getElementById('startDate').value,
        endDate: document.getElementById('endDate').value
    };
   
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
       
        const result = await response.json();
       
        if (result.status === 'success') {
            // Convert markdown to HTML using marked
            const htmlContent = marked.parse(result.response);
            results.innerHTML = `
                <div class="analysis-result">
                    ${htmlContent}
                </div>
            `;
            
            // Apply syntax highlighting to code blocks if any
            document.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightBlock(block);
            });
        } else {
            results.innerHTML = `<div class="alert alert-danger">Error: ${result.message}</div>`;
        }
    } catch (error) {
        results.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    } finally {
        loading.classList.add('d-none');
    }
});