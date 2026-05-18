window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Share to social media: copies the long-form post to the clipboard and opens
// the platform's compose window. We deliberately do NOT prefill the text via
// the URL: the post exceeds X (280) and Bluesky (300) character limits, so
// pre-filling would either truncate awkwardly or produce a massive URL.
// Users paste from the clipboard and trim per platform as needed.
const SHARE_POST_URL = "https://soda-inria.github.io/strable/";
const SHARE_POST_TEXT = `Real-world tables aren't just numbers. They're full of categories, names, codes, free text — and we've been benchmarking tabular ML as if those didn't exist.

So we built STRABLE (STRing-tABLE): the first large-scale empirical study of tabular learning with strings where we evaluate 445 pipelines on 108 real-world tables with raw strings.

Here's what we found:

🔤 Strings carry signal — numbers and strings are complementary, not redundant.

⚡ LLMs gain ground on free-text-heavy tables, while Simple encoders suffice when strings are categorical-dominant

📐 Dimensionality reduction matters for LLM embeddings — especially for decoder-only models

🌍 Rankings generalize — STRABLE's 108 datasets yield pipeline rankings close to the oracle.

📄 Paper: https://huggingface.co/papers/2605.12292
🤗 Dataset: https://huggingface.co/datasets/inria-soda/STRABLE-benchmark
🌐 Project page: https://soda-inria.github.io/strable/
💻 Code: https://github.com/soda-inria/strable

Huge thanks to my co-authors @Myung Jun Kim, @Félix Lefebvre, @Lennart Purucker, @Alan Arazi, @Eilam Shapira, @Roi Reichart, @Frank Hutter, @Marine Le Morvan, @David Holzmüller, @Gaël Varoquaux`;

function sharePost(event, platform) {
    if (event) event.preventDefault();

    // Always copy the full post to the clipboard.
    const copyPromise = (navigator.clipboard && navigator.clipboard.writeText)
        ? navigator.clipboard.writeText(SHARE_POST_TEXT).catch(() => {})
        : Promise.resolve();

    let shareUrl;
    if (platform === 'bluesky') {
        shareUrl = 'https://bsky.app/intent/compose';
    } else if (platform === 'twitter') {
        shareUrl = 'https://twitter.com/intent/tweet';
    } else if (platform === 'linkedin') {
        shareUrl = 'https://www.linkedin.com/sharing/share-offsite/?url=' +
                   encodeURIComponent(SHARE_POST_URL);
    } else {
        return;
    }

    const platformLabel = { bluesky: 'Bluesky', twitter: 'X', linkedin: 'LinkedIn' }[platform];

    copyPromise.finally(() => {
        showShareToast('Post copied — paste into the ' + platformLabel + ' editor (trim as needed for X/Bluesky)');
        window.open(shareUrl, '_blank', 'noopener,noreferrer');
    });
}

function showShareToast(message) {
    let toast = document.getElementById('share-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'share-toast';
        toast.setAttribute('role', 'status');
        toast.setAttribute('aria-live', 'polite');
        toast.style.cssText = [
            'position:fixed', 'bottom:24px', 'left:50%',
            'transform:translateX(-50%)', 'background:#1f2937',
            'color:#fff', 'padding:10px 18px', 'border-radius:8px',
            'z-index:9999', 'opacity:0', 'transition:opacity 0.25s',
            'font-size:14px', 'font-weight:500',
            'box-shadow:0 4px 12px rgba(0,0,0,0.15)',
            'max-width:90vw', 'text-align:center'
        ].join(';');
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    requestAnimationFrame(() => { toast.style.opacity = '1'; });
    clearTimeout(toast._hideTimer);
    toast._hideTimer = setTimeout(() => { toast.style.opacity = '0'; }, 2500);
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();
    
    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();

})
