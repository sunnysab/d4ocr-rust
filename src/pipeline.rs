use std::path::Path;

use image::imageops::FilterType;
use image::GrayImage;
use tract_onnx::prelude::Tensor;
use tract_onnx::prelude::*;

use super::MODEL_PATH;
use super::{GenericTransform, ImageTransform};
use super::{ImageSize, ResizeGrayImage, ToTensor};
use super::{ImageTransformResult, ToArray};

type TractSimplePlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct TransformationPipeline {
    steps: Vec<ImageTransform>,
    model: TractSimplePlan,
}

impl TransformationPipeline {
    pub fn new(image_size: ImageSize) -> Self {
        TransformationPipeline {
            steps: vec![
                ResizeGrayImage {
                    image_size: ImageSize {
                        width: image_size.width,
                        height: image_size.height,
                    },
                    filter: FilterType::CatmullRom,
                }
                .into(),
                ToArray {}.into(),
                ToTensor {}.into(),
            ],
            model: TransformationPipeline::load_model(&image_size),
        }
    }

    fn load_model(image_size: &ImageSize) -> TractSimplePlan {
        if !Path::new(MODEL_PATH).exists() {
            panic!("{MODEL_PATH} is not find");
        }
        let input_shape = tvec!(1, 1, image_size.height, image_size.width);
        let mut model = tract_onnx::onnx()
            .model_for_path(MODEL_PATH)
            .expect("Cannot read model")
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))
            .unwrap();
        model = model.with_output_names(vec!["output"]).unwrap();
        model.into_optimized().unwrap().into_runnable().unwrap()
    }

    fn transform_image(&self, image: GrayImage) -> Result<Tensor, &'static str> {
        let mut result = ImageTransformResult::GrayImage(image);

        for step in &self.steps {
            result = step.transform(result)?;
        }

        let to_tensor = ToTensor {};
        result = to_tensor.transform(result)?;

        match result {
            ImageTransformResult::Tensor(t) => Ok(t),
            _ => Err("Should be converted to tensor already"),
        }
    }

    fn extract_features(&self, image: GrayImage) -> Result<Vec<i64>, String> {
        let image_tensor = self.transform_image(image).expect("Cannot transform image");
        let result = self
            .model
            .run(tvec!(image_tensor))
            .expect("Cannot run model");
        let features: Vec<i64> = result[0]
            .to_array_view::<i64>()
            .expect("Cannot extract feature vector")
            .iter()
            .cloned()
            .collect();
        Ok(features)
    }

    pub fn recognize(&self, image: GrayImage) -> Result<String, String> {
        let vec = self.extract_features(image)?;

        let mut result = String::from("");
        let mut last_item: i64 = 0;
        for i in vec {
            if i == last_item {
                continue;
            } else {
                last_item = i
            }
            result.push_str(super::CHARSET[i as usize])
        }
        Ok(result)
    }
}
