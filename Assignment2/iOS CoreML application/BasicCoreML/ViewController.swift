

import UIKit

class ViewController: UIViewController, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    
    @IBOutlet weak var categoryLabel: UILabel!
    @IBOutlet weak var myImageView: UIImageView!
    
    let model = GoogLeNetPlaces()
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }
    
    @IBAction func importImage(_ sender: UIButton)
    {
        let image = UIImagePickerController()
        image.delegate = self
        
        image.sourceType = UIImagePickerControllerSourceType.photoLibrary
        
        image.allowsEditing = false
        
        self.present(image, animated: true)
        {
            
        }
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any])
    {
        if let image = info[UIImagePickerControllerOriginalImage] as? UIImage
        {
            myImageView.image = image
            let imageToAnalyse = image.renderResizedImage(newWidth: 74.6)
            if let sceneLabelString = sceneLabel(forImage: imageToAnalyse) {
                    categoryLabel.text = sceneLabelString
                }
        }
        else
        {
            //Error message
        }
        
        self.dismiss(animated: true, completion: nil)
    }

    
    func sceneLabel (forImage image:UIImage) -> String? {
        if let pixelBuffer = ImageProcessor.pixelBuffer(forImage: image.cgImage!) {
            guard let scene = try? model.prediction(sceneImage: pixelBuffer) else {fatalError("Unexpected runtime error")}
            return scene.sceneLabel
            
        }
        
        return nil
    }
    
    
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

extension UIImage {
    func renderResizedImage (newWidth: CGFloat) -> UIImage {
        
        let newSize = CGSize(width: newWidth, height: newWidth)
        
        let renderer = UIGraphicsImageRenderer(size: newSize)
        
        let image = renderer.image { (context) in
            self.draw(in: CGRect(origin: CGPoint(x: 0, y: 0), size: newSize))
        }
        return image
    }
}
