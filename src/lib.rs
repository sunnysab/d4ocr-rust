pub mod charset;

pub use charset::CHARSET;

use std::path::Path;
use enum_dispatch::enum_dispatch;
use image::imageops::{crop, resize, FilterType};
use image::GrayImage;
use tract_onnx::prelude::*;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use tract_onnx::prelude::{tract_ndarray::Array4, tract_ndarray::Ix4};
use tract_onnx::prelude::{tract_ndarray, Tensor};
use tract_onnx::tract_core::ndarray::Array;


#[enum_dispatch]
#[derive(Clone, Serialize, Deserialize)]
enum ImageTransform {
    ResizeGrayImage(ResizeGrayImage),
    ResizeGrayImageAspectRatio(ResizeGrayImageAspectRatio),
    CenterCrop(CenterCrop),
    Normalization(Normalization),
    Transpose(Transpose),
    ToArray(ToArray),
    ToTensor(ToTensor),
}

#[enum_dispatch(ImageTransform)]
trait GenericTransform {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str>;
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "FilterType")]
enum FilterOption {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

#[serde_as]
#[derive(Clone, Serialize, Deserialize)]
pub struct ResizeGrayImage {
    image_size: ImageSize,
    #[serde(with = "FilterOption")]
    filter: FilterType,
}

impl GenericTransform for ResizeGrayImage {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::GrayImage(image) => Ok(resize(
                &image,
                self.image_size.width as u32,
                self.image_size.height as u32,
                FilterType::CatmullRom,
            ).into()),
            ImageTransformResult::Tensor(_) => Err("Image resize not implemented for Tensor"),
            ImageTransformResult::Array4(_) => Err("Image resize not implemented for Array4"),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ResizeGrayImageAspectRatio {
    image_size: ImageSize,
    scale: f32,
    #[serde(with = "FilterOption")]
    filter: FilterType,
}

impl GenericTransform for ResizeGrayImageAspectRatio {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::GrayImage(image) => {
                let (height, width) = image.dimensions();
                let height = height as f32;
                let width = width as f32;
                let new_height = 100.0 * (self.image_size.height as f32) / self.scale;
                let new_width = 100.0 * (self.image_size.width as f32) / self.scale;

                let (final_height, final_width) = if height > width {
                    (new_width, new_height * height / width)
                } else {
                    (new_width * width / height, new_width)
                };

                Ok(resize(&image, final_width as u32, final_height as u32, self.filter).into())
            }
            ImageTransformResult::Tensor(_) => Err("Image resize not implemented for Tensor"),
            ImageTransformResult::Array4(_) => Err("Image resize not implemented for Array4"),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CenterCrop {
    crop_size: ImageSize,
}

impl GenericTransform for CenterCrop {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::GrayImage(image) => {
                let (height, width) = image.dimensions();
                let left = (width - self.crop_size.width as u32) / 2;
                let top = (height - self.crop_size.height as u32) / 2;
                let mut image_cropped = image;
                let image_cropped_new = crop(
                    &mut image_cropped,
                    top as u32,
                    left as u32,
                    self.crop_size.width as u32,
                    self.crop_size.height as u32,
                );
                Ok(image_cropped_new.to_image().into())
            }
            ImageTransformResult::Tensor(_) => Err("Image resize not implemented for Tensor"),
            ImageTransformResult::Array4(_) => Err("Image resize not implemented for Array4"),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Normalization {
    sub: [f32; 3],
    div: [f32; 3],
    zeroone: bool,
}

impl GenericTransform for Normalization {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::GrayImage(_) => Err("Not implemented"),
            ImageTransformResult::Tensor(_) => Err("Not implemented"),
            ImageTransformResult::Array4(arr) => {
                let sub = Array::from_shape_vec((1, 3, 1, 1), self.sub.to_vec())
                    .expect("Wrong conversion to array");
                let div = Array::from_shape_vec((1, 3, 1, 1), self.div.to_vec())
                    .expect("Wrong conversion to array");
                let new_arr = if self.zeroone {
                    (arr / 255.0 - sub) / div
                } else {
                    (arr - sub) / div
                };
                Ok(ImageTransformResult::Array4(new_arr))
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Transpose {
    axes: [usize; 4],
}

impl GenericTransform for Transpose {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::GrayImage(_) => Err("Not implemented"),
            ImageTransformResult::Array4(arr) => {
                let arr = arr.permuted_axes(self.axes);
                Ok(ImageTransformResult::Array4(arr))
            }
            ImageTransformResult::Tensor(tensor) => {
                // note that the same operation on Tensor is not safe as it is on Array4
                let tensor = tensor
                    .permute_axes(&self.axes)
                    .expect("Transpose should match the shape of the tensor");
                Ok(ImageTransformResult::Tensor(tensor))
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ToArray {}

impl GenericTransform for ToArray {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::GrayImage(image) => {
                let shape = image.dimensions();
                let arr = tract_ndarray::Array4::from_shape_fn(
                    (1_usize, 1_usize, shape.1 as usize, shape.0 as usize),
                    |(_, c, y, x)| ((image[(x as _, y as _)][c] as f32 / 255.) - 0.5) / 0.5,
                );
                Ok(ImageTransformResult::Array4(arr))
            }
            ImageTransformResult::Tensor(tensor) => {
                let dyn_arr = tensor
                    .into_array::<f32>()
                    .expect("Cannot convert tensor to Array4");
                let arr4 = dyn_arr
                    .into_dimensionality::<Ix4>()
                    .expect("Cannot convert dynamic Array to Array4");
                Ok(ImageTransformResult::Array4(arr4))
            }
            ImageTransformResult::Array4(arr4) => {
                // already an array
                Ok(ImageTransformResult::Tensor(arr4.into()))
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ToTensor {}

impl GenericTransform for ToTensor {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::GrayImage(image) => {
                let shape = image.dimensions();
                let tensor: Tensor = tract_ndarray::Array4::from_shape_fn(
                    (1_usize, 1_usize, shape.0 as usize, shape.1 as usize),
                    |(_, c, y, x)| image[(x as _, y as _)][c] as f32,
                )
                    .into();
                Ok(ImageTransformResult::Tensor(tensor))
            }
            ImageTransformResult::Tensor(tensor) => {
                // already a tensor
                Ok(ImageTransformResult::Tensor(tensor))
            }
            ImageTransformResult::Array4(arr4) => Ok(ImageTransformResult::Tensor(arr4.into())),
        }
    }
}

enum ImageTransformResult {
    GrayImage(GrayImage),
    Array4(Array4<f32>),
    Tensor(Tensor),
}

impl From<GrayImage> for ImageTransformResult {
    fn from(rgb_image: GrayImage) -> Self {
        ImageTransformResult::GrayImage(rgb_image)
    }
}

impl From<Tensor> for ImageTransformResult {
    fn from(tensor: Tensor) -> Self {
        ImageTransformResult::Tensor(tensor)
    }
}

type TractSimplePlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct TransformationPipeline {
    steps: Vec<ImageTransform>,
    model: TractSimplePlan,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ImageSize {
    pub width: usize,
    pub height: usize,
}


impl TransformationPipeline {
    pub fn new(image_size: ImageSize) -> Self{
        TransformationPipeline{
            steps: vec![
                ResizeGrayImage {
                    image_size: ImageSize {
                        width: image_size.width,
                        height: image_size.height,
                    },
                    filter: FilterType::CatmullRom,
                }.into(),
                ToArray {}.into(),
                ToTensor {}.into(),
            ],
            model: TransformationPipeline::load_model(&image_size),
        }
    }

    pub fn load_model(image_size: &ImageSize) -> TractSimplePlan {
        let name = "common.onnx";
        if !Path::new(name).exists(){
            println!("{} is not find", name);
            std::process::exit(-1)
        }
        let input_shape = tvec!(1, 1, image_size.height, image_size.width);
        let mut model = tract_onnx::onnx()
            .model_for_path(name)
            .expect("Cannot read model")
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))
            .unwrap();
        model = model.with_output_names(vec!["output"]).unwrap();
        model.into_optimized().unwrap().into_runnable().unwrap()
    }

    pub fn transform_image(&self, image: &GrayImage) -> Result<Tensor, &'static str> {
        let mut result = ImageTransformResult::GrayImage(image.clone());

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

    pub fn extract_features(&self, image: GrayImage) -> Result<Vec<i64>, String> {
        let image_tensor = self.transform_image(&image).expect("Cannot transform image");
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
}

