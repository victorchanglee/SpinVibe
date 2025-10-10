// Load content dynamically
document.addEventListener('DOMContentLoaded', function() {
    const mainContent = document.getElementById('main-content');
    const sidebarLinks = document.querySelectorAll('.sidebar a[data-section]');
    
    // Function to load content
    function loadContent(section) {
        // Show loading indicator
        mainContent.innerHTML = '<div class="loading">Loading...</div>';
        
        // Fetch the section HTML
        fetch(`sections/${section}.html`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Section not found');
                }
                return response.text();
            })
            .then(html => {
                mainContent.innerHTML = html;
                updateActiveLink(section);
                
                // Scroll to top when new content loads
                window.scrollTo(0, 0);
            })
            .catch(error => {
                mainContent.innerHTML = `
                    <div class="error">
                        <h1>Error</h1>
                        <p>Could not load the requested section: ${error.message}</p>
                    </div>
                `;
            });
    }
    
    // Update active link in sidebar
    function updateActiveLink(activeSection) {
        sidebarLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('data-section') === activeSection) {
                link.classList.add('active');
            }
        });
    }
    
    // Add click event listeners to sidebar links
    sidebarLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const section = this.getAttribute('data-section');
            loadContent(section);
            
            // Update URL hash without scrolling
            history.pushState(null, null, `#${this.getAttribute('href').substring(1)}`);
        });
    });
    
    // Load content based on URL hash on page load
    function loadInitialContent() {
        const hash = window.location.hash.substring(1);
        const validSections = ['overview', 'input', 'output', 'requirements', 'installation', 'example', 'changelog', 'license', 'theory','contributing'];
        
        if (hash && validSections.includes(hash)) {
            loadContent(hash);
        } else {
            // Load overview by default
            loadContent('overview');
        }
    }
    
    // Handle browser back/forward buttons
    window.addEventListener('popstate', loadInitialContent);
    
    // Load initial content
    loadInitialContent();
});
