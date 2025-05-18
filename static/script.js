import React, { useState } from 'react';

const ImageUploadForm = () => {
    const [result, setResult] = useState(null);
    const [features, setFeatures] = useState(null);
    const [showFeatures, setShowFeatures] = useState(false);

    const handleSubmit = async (event) => {
        event.preventDefault();
        const formData = new FormData();
        formData.append('image', event.target.image.files[0]);

        try {
            const response = await fetch('http://127.0.0.1:5000/process-image', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            setResult(data.result); // Assuming `result` comes from the backend
            setFeatures(data.features); // Assuming `features` is an array or object from the backend
        } catch (error) {
            console.error('Error processing the image:', error);
        }
    };

    const handleToggleFeatures = () => {
        setShowFeatures(!showFeatures);
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input type="file" name="image" accept="image/*" required />
                <button type="submit">Analyze</button>
            </form>

            {result && (
                <div>
                    <p>Result: <strong>{result}</strong></p>
                    <a href="#" onClick={handleToggleFeatures}>
                        {showFeatures ? 'Hide Extracted Features' : 'View Extracted Features'}
                    </a>
                    {showFeatures && features && (
                        <div>
                            <p>Extracted Features:</p>
                            <ul>
                                {Object.entries(features).map(([key, value], index) => (
                                    <li key={index}>
                                        {key}: {value}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default ImageUploadForm;
