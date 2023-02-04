use std::env;
use std::path::Path;

use d4ocr_rust::*;


fn main() {
    let args = env::args().nth(1).expect("no image path");
    let image = image::open(args).unwrap().to_luma8();
    let image_size = ImageSize{
        width: (image.width() as f32 * (64_f32 / image.height() as f32)) as usize,
        height: 64,
    };
    let _model =  TransformationPipeline::new(image_size);
    let res = _model.extract_features(image);
    let mut result = String::from("");
    let mut last_item:i64 = 0;
    for i in res.unwrap() {
        if i == last_item {
            continue
        }else{
            last_item = i
        }
        result.push_str(CHARSET[i as usize])
    }
    println!("---------------->{:?}",result);
}